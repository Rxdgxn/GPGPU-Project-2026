#include "physics_engine.h"
#include "virtual-world-physics/bounding_volume.h"
#include "virtual-world-physics/collision.h"
#include "virtual-world-physics/physics_gpu.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace physics {
void PhysicsEngine::ClearObjects() { m_objects.clear(); }

void PhysicsEngine::Update(float deltaTime) {
  // Reset statistics
  m_stats.detectedCollisions = 0;
  m_stats.objectCount = static_cast<int>(m_objects.size());

  auto startTime = std::chrono::high_resolution_clock::now();

  // Fixed timestep with accumulator
  m_accumulator += deltaTime;
  int subSteps = 0;
  while (m_accumulator >= m_fixedDeltaTime && subSteps < m_maxSubSteps) {
    ApplyGravity(m_fixedDeltaTime);

    for (auto &object : m_objects) {
      object.Integrate(m_fixedDeltaTime);
    }

    if (m_useGPU && m_gpuDetector) {
      try {
        auto collisions = m_gpuDetector->DetectCollisions(m_objects);

        m_stats.detectedCollisions = collisions.size();

        for (const auto &collision : collisions) {
          ResolveCollision(collision.indexA, collision.indexB, collision);
        }
      } catch (const std::exception &ex) {
        std::cerr << "CUDA collision backend failed, falling back to CPU: "
                  << ex.what() << std::endl;
        m_useGPU = false;
        BroadPhase();
        NarrowPhase();
      }
    } else {
      BroadPhase();

      // Multi pass narrow phase to properly distribute impulses (helps against the jitter)
      for (int i = 0; i < 8; i++) {
        NarrowPhase();
      }
    }

    m_accumulator -= m_fixedDeltaTime;
    subSteps++;
  }
  if (m_accumulator > m_fixedDeltaTime) {
    m_accumulator = 0.0f;
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  m_stats.collisionDetectionTime =
      std::chrono::duration<float, std::milli>(endTime - startTime).count();
}

void PhysicsEngine::ApplyGravity(float deltaTime) {
  for (auto& obj : m_objects) {
    obj.ApplyForce(m_gravity * obj.mass);
  }
}

void PhysicsEngine::BroadPhase() {
  m_possiblePairs = GetPotentialCollisionPairs();
}


CollisionPairs PhysicsEngine::GetPotentialCollisionPairs() {
  CollisionPairs ret;

  // TODO: maybe change the destructor so there isn't an extra pass each frame for transforming objects to leaves
  const size_t n = m_objects.size();
  auto leaves = new BVHNode* [n];

  for (size_t i = 0; i < n; i++) {
    leaves[i] = new BVHNode(&m_objects[i], i);
  }

  auto root = new BVHNode(leaves, n);

  for (size_t i = 0; i < n; i++) {
    std::vector<size_t> neighbors;
    getCollisions(root, m_objects[i].boundingVolume, neighbors);

    for (size_t j : neighbors) {
      if (i < j) {
        ret.push_back({i, j});
      }
    }
  }

  delete root;

  return ret;
}

void PhysicsEngine::NarrowPhase() {
  for (auto& [i, j] : m_possiblePairs) {
    auto col = DetectCollision(i, j);

    // This is technically redundant while using AABBs, since the objects are the AABBs themselves
    if (col.isValid) {
      m_stats.detectedCollisions++;
      ResolveCollision(i, j, col);
    }
  }
}

CollisionInfo PhysicsEngine::DetectCollision(size_t indexA, size_t indexB) {
  return ComputeBoxBoxCollision(indexA, indexB, m_objects[indexA].boundingVolume, m_objects[indexB].boundingVolume);
}

CollisionInfo PhysicsEngine::ComputeBoxBoxCollision(size_t indexA, size_t indexB,
                                      const bounding_volume_t &boxA,
                                      const bounding_volume_t &boxB) {
  CollisionInfo col;
  col.indexA = indexA;
  col.indexB = indexB;

  auto delta = boxB.center - boxA.center;
  auto overlap = (boxA.sizes + boxB.sizes) - glm::abs(delta);

  if (overlap.x < overlap.y && overlap.x < overlap.z) {
    col.normal = glm::vec3(delta.x < 0.0f ? -1.0f : 1.0f, 0.0f, 0.0f);
    col.penetration = overlap.x;
  } else if (overlap.y < overlap.z) {
    col.normal = glm::vec3(0.0f, delta.y < 0.0f ? -1.0f : 1.0f, 0.0f);
    col.penetration = overlap.y;
  } else {
    col.normal = glm::vec3(0.0f, 0.0f, delta.z < 0.0f ? -1.0f : 1.0f);
    col.penetration = overlap.z;
  }

  col.isValid = true;
  return col;
}

void PhysicsEngine::ResolveCollision(size_t indexA, size_t indexB,
                                     const CollisionInfo &collision) {

  PhysicsObject &A = m_objects[indexA];
  PhysicsObject &B = m_objects[indexB];

  float invMassA = A.isStatic ? 0.0f : 1.0f / A.mass;
  float invMassB = B.isStatic ? 0.0f : 1.0f / B.mass;
  float invMassSum = invMassA + invMassB;

  if (invMassSum == 0.0f) return;

  auto& normal = collision.normal;

  // 1. Separation (positional correction)
  float slop = 0.001f; // Slop threshold prevents jitter from tiny sub-millimeter overlaps
  float percent = 0.8f; // Percentage of overlap to fix per step
  float correctedPenetration = std::max(collision.penetration - slop, 0.0f) * percent;

  auto correction = correctedPenetration / invMassSum * normal;
  
  A.position -= correction * invMassA;
  B.position += correction * invMassB;

  A.UpdateBoundingVolume();
  B.UpdateBoundingVolume();

  // 2. Bounce (normal impulse)
  auto relativeVelocity = B.velocity - A.velocity;
  float velAlongNormal = glm::dot(relativeVelocity, normal);

  // If they are already moving apart, no bounce
  if (velAlongNormal > 0.0f) {
    return;
  }

  float e = std::min(A.restitution, B.restitution);

  // Threshold to stop micro-bouncing on flat surfaces
  if (std::abs(velAlongNormal) < 0.2f) {
    e = 0.0f;
  }

  float j = -(1.0f + e) * velAlongNormal / invMassSum;
  auto normalImpulse = j * normal;

  A.ApplyImpulse(-normalImpulse);
  B.ApplyImpulse(normalImpulse);

  // 3. Friction impulse
  relativeVelocity = B.velocity - A.velocity;
  velAlongNormal = glm::dot(relativeVelocity, normal);

  auto tangentVelocity = relativeVelocity - velAlongNormal * normal;

  if (glm::dot(tangentVelocity, tangentVelocity) < 1e-8f) {
    return;
  }
  
  auto tangent = glm::normalize(tangentVelocity);
  float jt = -glm::dot(relativeVelocity, tangent) / invMassSum;

  float mu = 0.5f * (A.friction + B.friction);
  jt = std::clamp(jt, -j * mu, j * mu);

  auto frictionImpulse = jt * tangent;

  A.ApplyImpulse(-frictionImpulse);
  B.ApplyImpulse(frictionImpulse);
}

void PhysicsEngine::Init(const glm::vec3 &gravity, bool useGpu) {
  m_gravity = gravity;
  m_useGPU = false;

  delete m_gpuDetector;
  m_gpuDetector = nullptr;

  if (useGpu) {
    m_gpuDetector = new GPUCollisionDetector();
    if (m_gpuDetector->Initialize()) {
      m_useGPU = true;
    } else {
      delete m_gpuDetector;
      m_gpuDetector = nullptr;
      std::cerr << "CUDA collision backend unavailable; using CPU backend."
                << std::endl;
    }
  }
}

PhysicsEngine::~PhysicsEngine() { delete m_gpuDetector; }
} // namespace physics
