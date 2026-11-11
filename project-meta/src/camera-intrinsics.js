const ASSUMED_HORIZONTAL_FOV_DEGREES = 60

export function createCameraIntrinsics(width, height) {
  const horizontalFovRadians =
    (ASSUMED_HORIZONTAL_FOV_DEGREES * Math.PI) / 180
  const focalLength = width / (2 * Math.tan(horizontalFovRadians / 2))

  return {
    fx: focalLength,
    fy: focalLength,
    cx: width / 2,
    cy: height / 2,
  }
}
