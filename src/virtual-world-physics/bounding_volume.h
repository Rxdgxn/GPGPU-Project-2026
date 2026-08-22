/*
  * Bounding volume representation for physics computations.
  * Simple axis-aligned bounding box (AABB) representation, defined by a center point and half-extents along each axis.
  * You may also use Oriented Bounding Boxes (OBB) or other bounding volume types, if you want. :)
 */
#pragma once

#include <glm/glm.hpp>

namespace physics {

typedef struct bounding_volume_t_ {
  glm::vec3 center{};
  glm::vec3 sizes{0.5f};
} bounding_volume_t;

inline bounding_volume_t encompassVolumes(const bounding_volume_t& a, const bounding_volume_t& b) {
    auto minA = a.center - a.sizes;
    auto maxA = a.center + a.sizes;
    auto minB = b.center - b.sizes;
    auto maxB = b.center + b.sizes;

    auto minC = min(minA, minB);
    auto maxC = max(maxA, maxB);

    return {(minC + maxC) * 0.5f, (maxC - minC) * 0.5f};
}

} // namespace physics
