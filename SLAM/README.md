## Robot Motion Model, LiDAR Landmark Detection, and Landmark Matching

A velocity-based motion model for a point-mass robot, a LiDAR sensor model that detects landmarks from scan gradients, and nearest-neighbor matching of detected landmarks to ground truth.

### Part 1: Velocity Motion Model

Given the pose (xₜ₋₁, yₜ₋₁, θₜ₋₁) and the commanded velocities (v, v_θ), predict the next pose. Let Δt be the time step.

**Case 1: v > 0 and v_θ > 0** (the robot moves along a circular arc). The local displacement is the chord of the arc:

```
α  = v_θ · Δt              (angle of rotation)
R  = v · Δt / α            (radius of rotation)
d  = R · sin(α/2)          (half the chord length)
Δx = 2d · cos(α/2)         (local x displacement)
Δy = 2d · sin(α/2)         (local y displacement)
```

Δx and Δy are in the robot's local frame, so they are rotated by the previous heading θₜ₋₁ before being added:

```
xₜ = xₜ₋₁ + Δx·cos(θₜ₋₁) − Δy·sin(θₜ₋₁)
yₜ = yₜ₋₁ + Δx·sin(θₜ₋₁) + Δy·cos(θₜ₋₁)
θₜ = θₜ₋₁ + α
```

**Case 2: v > 0 and v_θ = 0** (straight line). Here α = 0, Δx = v·Δt, and Δy = 0:

```
xₜ = xₜ₋₁ + v·Δt·cos(θₜ₋₁)
yₜ = yₜ₋₁ + v·Δt·sin(θₜ₋₁)
θₜ = θₜ₋₁
```

Noise in the translational and angular velocity readings accumulates over time, so the predicted path drifts from the ground-truth path.

### Part 2: LiDAR Sensor Model and Landmark Detection

1. Convert each scan to global coordinates by adding the robot's heading to every beam angle, then offsetting by the robot's (x, y).
2. Compute the gradient of the range readings across beam indices and keep only indices with large gradients (threshold 0.15; a threshold of 0.24 found fewer landmarks).
3. Use a stack to pair each negative gradient (landmark start) with the next positive gradient (landmark end). Because a LiDAR scan covers a full 2π, gradients near the end of the scan can pair with ones at the start.
4. For each pair, average the global angles and the ranges (plus a 0.15 offset) to place the landmark.

| Time step | Landmarks detected | Notes |
|---|---|---|
| t = 0 | 5 | Robot at (1, 0), θ = 0 |
| t = 15 | 5 | Landmarks found in roughly the same locations |
| t = 51 | 6 (matches ground truth) | Robot moved closer to a landmark that was previously out of range |

### Part 3: Landmark Matching

Each detected landmark is paired with its nearest ground-truth landmark. If the nearest one is more than 1.0 away, the detection is not paired. More landmarks are detected at t = 51, so more pairs are formed there.



## Feature-Based and Featureless Localization

Corrects the drifting motion-model pose using LiDAR observations, in two ways: **feature-based** (landmark pairs and a similarity transform) and **featureless** (ICP against the walls).

### Part 1: Feature-Based Localization

Given matched pairs of detected landmarks lᵢ and ground-truth landmarks rᵢ (i = 1..m), find the similarity transform that maps one set onto the other:

```
rᵢ = λ·R·lᵢ + t
```

where λ is a scale, R is a 2D rotation, and t is a translation. The transform minimizes the total squared error:

```
minimize  Σᵢ ‖λ·R·lᵢ + t − rᵢ‖²
```

**Derivation sketch**

1. Center both point sets on their centroids: l̄ = (1/m)·Σlᵢ, r̄ = (1/m)·Σrᵢ, l'ᵢ = lᵢ − l̄, r'ᵢ = rᵢ − r̄.
2. In centered coordinates the cost splits into a rotation/scale term and a translation term, because the cross term vanishes (Σl'ᵢ = Σr'ᵢ = 0):

```
Σ ‖λRl'ᵢ − r'ᵢ‖²  +  m·‖t'‖²,     where t' = λR·l̄ + t − r̄
```

3. The second term is minimized by t' = 0, so t = r̄ − λR·l̄.
4. The scale is estimated from the ratio of spreads (‖Rl'ᵢ‖ = ‖l'ᵢ‖, so R is not needed):

```
λ = sqrt( Σ‖r'ᵢ‖² / Σ‖l'ᵢ‖² )
```

5. The rotation angle α comes from the centered points:

```
cos α = Σ(r'ₓl'ₓ + r'ᵧl'ᵧ) / N
sin α = Σ(−r'ₓl'ᵧ + r'ᵧl'ₓ) / N
```

where N is the norm of the vector (Σ(r'ₓl'ₓ + r'ᵧl'ᵧ), Σ(−r'ₓl'ᵧ + r'ᵧl'ₓ)).

To correct a point p and the heading θ:

```
p'    = λ·R·p + t
θ'    = θ + α
```

**Procedure at each time step:** predict the pose with the motion model, re-detect and re-pair landmarks, compute the similarity transform, and correct the pose. Previous corrections carry forward, so motion-model error does not accumulate.

**Result:** the corrected trajectory almost perfectly overlaps the ground-truth path.

### Part 2: Featureless Localization (ICP)

Instead of landmarks, align the scan directly against the wall geometry:

1. Subsample about 36 LiDAR rays and compute their endpoints.
2. For each endpoint, find the closest wall point; keep the pair only if it is within 0.1.
3. Compute a transform from these pairs as in Part 1, with `fix_scale=True`, since the motion is rigid.
4. Apply it, and repeat 100 times on the same time step (Iterative Closest Point), composing the transforms into one combined correction.
5. Carry the correction forward to later time steps, as in Part 1.

**Results**

- Both methods track the ground truth far better than the motion model alone.
- Featureless ICP performed slightly better than the feature-based method, likely because wall data is available at almost every step (many of the ~36 rays hit walls), while there were only 6 landmarks.
- Aligning overall wall geometry makes ICP less sensitive to individual detection and pairing errors.
