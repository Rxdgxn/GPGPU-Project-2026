#pragma once

#include "bounding_volume.h"
#include <glm/glm.hpp>

namespace physics {

inline bool Intersects(const bounding_volume_t &boxA,
                       const bounding_volume_t &boxB) {
    // Simple AABB intersection test
    return boxA.center.x - boxA.sizes.x <= boxB.center.x + boxB.sizes.x &&
           boxA.center.x + boxA.sizes.x >= boxB.center.x - boxB.sizes.x &&
           boxA.center.y - boxA.sizes.y <= boxB.center.y + boxB.sizes.y &&
           boxA.center.y + boxA.sizes.y >= boxB.center.y - boxB.sizes.y &&
           boxA.center.z - boxA.sizes.z <= boxB.center.z + boxB.sizes.z &&
           boxA.center.z + boxA.sizes.z >= boxB.center.z - boxB.sizes.z;
}

} // namespace physics
