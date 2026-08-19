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
      NarrowPhase();
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
  // TODO
  for (auto& obj : m_objects) {
    obj.ApplyForce(m_gravity * obj.mass * deltaTime);
  }
}

void PhysicsEngine::BroadPhase() {
  // TODO
  m_possiblePairs = GetPotentialCollisionPairs();
}


CollisionPairs PhysicsEngine::GetPotentialCollisionPairs() {
  // TODO
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
  // TODO
  for (auto& [i, j] : m_possiblePairs) {
    auto col = DetectCollision(i, j);

    if (col.isValid) {
      m_stats.detectedCollisions++;
      ResolveCollision(i, j, col);
    }
  }
}

CollisionInfo PhysicsEngine::DetectCollision(size_t indexA, size_t indexB) {
  // TODO
  return ComputeBoxBoxCollision(indexA, indexB, m_objects[indexA].boundingVolume, m_objects[indexB].boundingVolume);
}

CollisionInfo
PhysicsEngine::ComputeBoxBoxCollision(size_t indexA, size_t indexB,
                                      const bounding_volume_t &boxA,
                                      const bounding_volume_t &boxB) {
  // TODO
  CollisionInfo col;
  col.isValid = true;

  return col;
}

void PhysicsEngine::ResolveCollision(size_t indexA, size_t indexB,
                                     const CollisionInfo &collision) {
  // TODO
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
