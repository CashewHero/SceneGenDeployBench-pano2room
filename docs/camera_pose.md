# Camera Pose Inputs

`camera_pose` is the one `sample.data` value that may be a JSON object instead of a file path. Other common inputs such as `image`, `depth`, `scene`, and `mesh` are paths.

Example job request fragment:

```json
{
  "sample": {
    "data": {
      "image": "/data/datasets/example/image/0001.png",
      "camera_pose": {
        "position": [0.0, 0.0, 0.0],
        "rotation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
      }
    },
    "metadata": {
      "pose_convention": "camera_to_world",
      "pose_coordinate_system": "NED",
      "pose_units": "meters",
      "projection": "equirectangular"
    }
  }
}
```

## Contract

`camera_pose` fields:

- `position`: `[x, y, z]`; defaults to `[0.0, 0.0, 0.0]`.
- `rotation_quaternion_xyzw`: `[qx, qy, qz, qw]`; defaults to identity `[0.0, 0.0, 0.0, 1.0]`.

Supported metadata `pose_convention` values:

- `camera_to_world`: transform maps camera-local coordinates into world coordinates.
- `world_to_camera`: transform maps world coordinates into camera-local coordinates.

Supported metadata `pose_coordinate_system` values:

- `NED`: `+X` north, `+Y` east, `+Z` down.
- `ENU`: `+X` east, `+Y` north, `+Z` up.
- `RDF`: `+X` right, `+Y` down, `+Z` forward.
- `RUB`: `+X` right, `+Y` up, `+Z` backward.

Supported metadata `pose_units` values:

- `meters`; if absent, units are unspecified

## Runner Handling

`camera_pose` contains only frame-specific pose values. Read pose context from `sample.metadata`, such as `pose_convention`, `pose_coordinate_system` or `pose_units`.

Built-in pose defaults are zero `position`, identity `rotation_quaternion_xyzw`, and `camera_to_world` convention when no `pose_convention` is provided. Missing `pose_coordinate_system` or `pose_units` means unspecified.

Runner wrappers should convert pose values as needed for the downstream model. Runners should tolerate unspecified fields unless they are actually needed.
