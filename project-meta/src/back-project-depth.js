const DEFAULT_NEAR_DEPTH = 0.6
const DEFAULT_FAR_DEPTH = 3.0

function getDepthDimensions(depthMap) {
  return {
    width: depthMap.width ?? depthMap.dims?.at(-1),
    height: depthMap.height ?? depthMap.dims?.at(-2),
  }
}

export function backProjectDepthTo3D(
  depthMap,
  mask,
  intrinsics,
  {
    nearDepth = DEFAULT_NEAR_DEPTH,
    farDepth = DEFAULT_FAR_DEPTH,
    positions,
    sourcePixelIndices,
  } = {},
) {
  const depthValues = depthMap.data ?? depthMap
  const maskValues = mask.data ?? mask
  const { width, height } = getDepthDimensions(depthMap)
  const maskWidth = mask.width ?? width
  const maskHeight = mask.height ?? height
  const maxPointCount = width * height
  const outputPositions =
    positions ?? new Float32Array(maxPointCount * 3)
  const outputSourcePixelIndices =
    sourcePixelIndices ?? new Uint32Array(maxPointCount)

  if (!width || !height || depthValues.length < maxPointCount) {
    throw new Error('Depth map dimensions do not match its data.')
  }

  if (outputPositions.length < maxPointCount * 3) {
    throw new Error('The position buffer is too small for the depth map.')
  }

  if (outputSourcePixelIndices.length < maxPointCount) {
    throw new Error('The source-index buffer is too small for the depth map.')
  }

  let minInverseDepth = Infinity
  let maxInverseDepth = -Infinity

  for (let y = 0; y < height; y += 1) {
    const maskY = Math.min(
      Math.floor((y / height) * maskHeight),
      maskHeight - 1,
    )

    for (let x = 0; x < width; x += 1) {
      const maskX = Math.min(
        Math.floor((x / width) * maskWidth),
        maskWidth - 1,
      )
      const maskIndex = maskY * maskWidth + maskX
      const inverseDepth = depthValues[y * width + x]

      if (maskValues[maskIndex] !== 1 || !Number.isFinite(inverseDepth)) {
        continue
      }

      minInverseDepth = Math.min(minInverseDepth, inverseDepth)
      maxInverseDepth = Math.max(maxInverseDepth, inverseDepth)
    }
  }

  if (!Number.isFinite(minInverseDepth)) {
    return {
      positions: outputPositions,
      sourcePixelIndices: outputSourcePixelIndices,
      count: 0,
    }
  }

  const inverseDepthRange = maxInverseDepth - minInverseDepth || 1
  const videoWidth = intrinsics.cx * 2
  const videoHeight = intrinsics.cy * 2
  let pointCount = 0

  for (let y = 0; y < height; y += 1) {
    const maskY = Math.min(
      Math.floor((y / height) * maskHeight),
      maskHeight - 1,
    )

    for (let x = 0; x < width; x += 1) {
      const maskX = Math.min(
        Math.floor((x / width) * maskWidth),
        maskWidth - 1,
      )
      const maskIndex = maskY * maskWidth + maskX

      if (maskValues[maskIndex] !== 1) continue

      const sourceIndex = y * width + x
      const inverseDepth = depthValues[sourceIndex]

      if (!Number.isFinite(inverseDepth)) continue

      const normalizedInverseDepth =
        (inverseDepth - minInverseDepth) / inverseDepthRange
      const zDistance =
        farDepth - normalizedInverseDepth * (farDepth - nearDepth)
      const videoX = ((x + 0.5) / width) * videoWidth
      const videoY = ((y + 0.5) / height) * videoHeight
      const positionIndex = pointCount * 3

      outputPositions[positionIndex] =
        ((videoX - intrinsics.cx) * zDistance) / intrinsics.fx
      outputPositions[positionIndex + 1] =
        (-(videoY - intrinsics.cy) * zDistance) / intrinsics.fy
      outputPositions[positionIndex + 2] = -zDistance
      outputSourcePixelIndices[pointCount] = sourceIndex
      pointCount += 1
    }
  }

  return {
    positions: outputPositions,
    sourcePixelIndices: outputSourcePixelIndices,
    count: pointCount,
  }
}
