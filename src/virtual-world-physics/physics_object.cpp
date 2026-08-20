#include "physics_object.h"
#include <glm/gtc/matrix_transform.hpp>

namespace physics {
glm::mat4 PhysicsObject::GetModelMatrix() const {
  glm::mat4 modelMatrix = glm::mat4(1.0f);
  modelMatrix = glm::translate(modelMatrix, position);
  modelMatrix = glm::scale(modelMatrix, scale);
  return modelMatrix;
}

void PhysicsObject::UpdateBoundingVolume() {
  boundingVolume.center = position;
}

void PhysicsObject::ApplyForce(const glm::vec3 &force) {
  // F = m * a => a = F / m
  if (!isStatic)
    acceleration += force / mass;
}

void PhysicsObject::ApplyImpulse(const glm::vec3 &impulse) {
  // p = m * v => v = p / m
  if (!isStatic)
    velocity += impulse / mass;
}

void PhysicsObject::Integrate(float deltaTime) {
  if (!isStatic) {
    velocity += acceleration * deltaTime;
    position += velocity * deltaTime;

    UpdateBoundingVolume();
  }

  acceleration = glm::vec3(0.0f);
}
} // namespace physics
