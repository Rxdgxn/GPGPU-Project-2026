#pragma once

#include "bounding_volume.h"
#include "physics_object.h"
#include <glm/glm.hpp>

namespace physics {

// "merging" 2 volumes together
inline bounding_volume_t encompassVolumes(const bounding_volume_t& a, const bounding_volume_t& b) {
    auto minA = a.center - a.sizes;
    auto maxA = a.center + a.sizes;
    auto minB = b.center - b.sizes;
    auto maxB = b.center + b.sizes;

    auto minC = min(minA, minB);
    auto maxC = max(maxA, maxB);

    return {(minC + maxC) * 0.5f, (maxC - minC) * 0.5f};
}

struct BVHNode {
    bounding_volume_t aabb;
    PhysicsObject *obj = nullptr; // non null only for leaves
    BVHNode *left = nullptr;
    BVHNode *right = nullptr;
    size_t index = -1; // to work faster with the api

    bool isLeaf() {
        return left == nullptr && right == nullptr;
    }

    ~BVHNode() {
        if (left == right) {
            delete left;
        }
        else {
            delete left;
            delete right;
        }
    }

    BVHNode(PhysicsObject *o, size_t idx) {
        obj = o;
        aabb = o->boundingVolume;
        index = idx;
    }

    // Note: this is mostly stolen from the workshop
    BVHNode(BVHNode* nodes[], int n) {
        int axis = rand() % 3;

        if (n == 1) {
            left = right = nodes[0];
            aabb = left->aabb;
        }
        else if (n == 2) {
            if (nodes[0]->aabb.center[axis] < nodes[1]->aabb.center[axis]) {
                left = nodes[0];
                right = nodes[1];
            }
            else {
                left = nodes[1];
                right = nodes[0];
            }

            aabb = encompassVolumes(left->aabb, right->aabb);
        }
        else {
            std::sort(nodes, nodes + n, [axis](BVHNode *a, BVHNode *b) {
                return a->aabb.center[axis] < b->aabb.center[axis];
            });

            int mid = n / 2;
            left = new BVHNode(nodes, mid);
            right = new BVHNode(nodes + mid, n - mid);
            aabb = encompassVolumes(left->aabb, right->aabb);
        }
    }
};

} // namespace physics