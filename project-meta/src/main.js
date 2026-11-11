import * as THREE from 'three'
import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision'
import { pipeline } from '@huggingface/transformers'
import { backProjectDepthTo3D } from './back-project-depth.js'
import { createCameraIntrinsics } from './camera-intrinsics.js'
import './style.css'

const MEDIAPIPE_WASM_ROOT =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/wasm'
const SELFIE_SEGMENTER_MODEL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite'
const DEPTH_MODEL = 'onnx-community/depth-anything-v2-small'

const scene = new THREE.Scene()
scene.background = new THREE.Color(0x0b0d12)

const camera = new THREE.PerspectiveCamera(
  75,
  window.innerWidth / window.innerHeight,
  0.1,
  100,
)
camera.position.set(0, 0, 0)

const renderer = new THREE.WebGLRenderer({ antialias: true })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.setSize(window.innerWidth, window.innerHeight)
document.querySelector('#app').appendChild(renderer.domElement)

const cube = new THREE.Mesh(
  new THREE.BoxGeometry(1, 1, 1),
  new THREE.MeshNormalMaterial(),
)
cube.position.set(1.5, 0, -3)
scene.add(cube)

const pointCloudGeometry = new THREE.BufferGeometry()
const pointCloudMaterial = new THREE.PointsMaterial({
  size: 0.01,
  sizeAttenuation: true,
  vertexColors: true,
})
const pointCloud = new THREE.Points(pointCloudGeometry, pointCloudMaterial)
pointCloud.frustumCulled = false
scene.add(pointCloud)

let pointCloudPositions = new Float32Array()
let pointCloudColors = new Float32Array()
let pointCloudSourcePixelIndices = new Uint32Array()
let allPixelsMask = new Uint8Array()

function ensurePointCloudCapacity(maxPointCount) {
  if (pointCloudSourcePixelIndices.length >= maxPointCount) return

  pointCloudPositions = new Float32Array(maxPointCount * 3)
  pointCloudColors = new Float32Array(maxPointCount * 3)
  pointCloudSourcePixelIndices = new Uint32Array(maxPointCount)
  pointCloudGeometry.setAttribute(
    'position',
    new THREE.BufferAttribute(pointCloudPositions, 3),
  )
  pointCloudGeometry.setAttribute(
    'color',
    new THREE.BufferAttribute(pointCloudColors, 3),
  )
  pointCloudGeometry.setDrawRange(0, 0)
}

function updatePointCloudGeometry(predictedDepth) {
  const width = predictedDepth.dims.at(-1)
  const height = predictedDepth.dims.at(-2)
  const maxPointCount = width * height

  ensurePointCloudCapacity(maxPointCount)

  if (allPixelsMask.length !== maxPointCount) {
    allPixelsMask = new Uint8Array(maxPointCount)
    allPixelsMask.fill(1)
  }

  const { count } = backProjectDepthTo3D(
    predictedDepth,
    { data: allPixelsMask, width, height },
    createCameraIntrinsics(video.videoWidth, video.videoHeight),
    {
      positions: pointCloudPositions,
      sourcePixelIndices: pointCloudSourcePixelIndices,
    },
  )

  pointCloudGeometry.attributes.position.needsUpdate = true
  pointCloudGeometry.setDrawRange(0, count)

  return { count, width, height }
}

function updatePointCloudColors(pointCount, depthWidth, depthHeight) {
  const frameWidth = depthInputCanvas.width
  const frameHeight = depthInputCanvas.height
  const frame = depthInputContext.getImageData(
    0,
    0,
    frameWidth,
    frameHeight,
  )

  for (let pointIndex = 0; pointIndex < pointCount; pointIndex += 1) {
    const sourceIndex = pointCloudSourcePixelIndices[pointIndex]
    const depthX = sourceIndex % depthWidth
    const depthY = Math.floor(sourceIndex / depthWidth)
    const frameX = Math.min(
      Math.floor(((depthX + 0.5) / depthWidth) * frameWidth),
      frameWidth - 1,
    )
    const frameY = Math.min(
      Math.floor(((depthY + 0.5) / depthHeight) * frameHeight),
      frameHeight - 1,
    )
    const frameIndex = (frameY * frameWidth + frameX) * 4
    const colorIndex = pointIndex * 3

    pointCloudColors[colorIndex] = frame.data[frameIndex] / 255
    pointCloudColors[colorIndex + 1] = frame.data[frameIndex + 1] / 255
    pointCloudColors[colorIndex + 2] = frame.data[frameIndex + 2] / 255
  }

  pointCloudGeometry.attributes.color.needsUpdate = true
}

function animate() {
  cube.rotation.x += 0.01
  cube.rotation.y += 0.01

  renderer.render(scene, camera)
}

renderer.setAnimationLoop(animate)

const video = document.createElement('video')
video.autoplay = true
video.muted = true
video.playsInline = true
video.hidden = true
video.setAttribute('aria-hidden', 'true')
document.body.appendChild(video)

const visionOverlays = document.createElement('div')
visionOverlays.className = 'vision-overlays'
document.body.appendChild(visionOverlays)

const depthStatus = document.createElement('p')
depthStatus.className = 'depth-status'
depthStatus.setAttribute('role', 'status')
depthStatus.setAttribute('aria-live', 'polite')
depthStatus.textContent = 'Loading Depth Anything V2…'
document.body.appendChild(depthStatus)

const webcamCanvas = document.createElement('canvas')
webcamCanvas.className = 'webcam-overlay'
webcamCanvas.width = 320
webcamCanvas.height = 240
visionOverlays.appendChild(webcamCanvas)

const webcamContext = webcamCanvas.getContext('2d')

const depthDebugCanvas = document.createElement('canvas')
depthDebugCanvas.className = 'depth-debug'
depthDebugCanvas.width = 320
depthDebugCanvas.height = 240
depthDebugCanvas.title = 'Press D to toggle the depth debug view'
visionOverlays.appendChild(depthDebugCanvas)

const depthDebugContext = depthDebugCanvas.getContext('2d')
let imageSegmenter
let lastSegmentedVideoTime = -1

async function loadImageSegmenter() {
  try {
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_ROOT)
    imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
      baseOptions: { modelAssetPath: SELFIE_SEGMENTER_MODEL },
      runningMode: 'VIDEO',
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    })
  } catch (error) {
    showCameraMessage('Selfie segmentation is unavailable.')
    console.error('Unable to load selfie segmentation:', error)
  }
}

function renderWebcamFrame() {
  if (imageSegmenter && video.currentTime !== lastSegmentedVideoTime) {
    lastSegmentedVideoTime = video.currentTime
    imageSegmenter.segmentForVideo(video, performance.now(), (result) => {
      const categoryMask = result.categoryMask

      if (!categoryMask) {
        webcamContext.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height)
        return
      }

      webcamContext.drawImage(
        video,
        0,
        0,
        webcamCanvas.width,
        webcamCanvas.height,
      )

      const frame = webcamContext.getImageData(
        0,
        0,
        webcamCanvas.width,
        webcamCanvas.height,
      )
      const mask = categoryMask.getAsUint8Array()

      for (let y = 0; y < webcamCanvas.height; y += 1) {
        const maskY = Math.floor((y / webcamCanvas.height) * categoryMask.height)

        for (let x = 0; x < webcamCanvas.width; x += 1) {
          const maskX = Math.floor((x / webcamCanvas.width) * categoryMask.width)
          const maskIndex = maskY * categoryMask.width + maskX
          const alphaIndex = (y * webcamCanvas.width + x) * 4 + 3

          frame.data[alphaIndex] = mask[maskIndex] === 1 ? 255 : 0
        }
      }

      webcamContext.putImageData(frame, 0, 0)
    })
  } else if (!imageSegmenter) {
    webcamContext.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height)
  }

  if ('requestVideoFrameCallback' in video) {
    video.requestVideoFrameCallback(renderWebcamFrame)
  } else {
    requestAnimationFrame(renderWebcamFrame)
  }
}

loadImageSegmenter()

async function loadDepthModel() {
  const loadForBackend = (device, label) => {
    depthStatus.textContent = `Loading Depth Anything V2 (${label})…`

    return pipeline('depth-estimation', DEPTH_MODEL, {
      device,
      progress_callback: (progress) => {
        if (progress.status === 'progress' && Number.isFinite(progress.progress)) {
          depthStatus.textContent = `Loading Depth Anything V2 (${label})… ${Math.round(progress.progress)}%`
        }
      },
    })
  }

  try {
    if ('gpu' in navigator) {
      try {
        const depthEstimator = await loadForBackend('webgpu', 'WebGPU')
        activeDepthBackend = 'WebGPU'
        depthStatus.textContent = `Depth model ready · ${activeDepthBackend}`
        return depthEstimator
      } catch (error) {
        console.warn('WebGPU depth initialization failed; using WASM:', error)
        depthStatus.textContent = 'WebGPU unavailable · falling back to WASM…'
      }
    }

    const depthEstimator = await loadForBackend('wasm', 'WASM')
    activeDepthBackend = 'WASM'
    depthStatus.textContent = `Depth model ready · ${activeDepthBackend}`
    return depthEstimator
  } catch (error) {
    activeDepthBackend = 'Unavailable'
    depthStatus.textContent = 'Depth model failed to load'
    console.error('Unable to load Depth Anything V2:', error)
    return null
  }
}

let activeDepthBackend = 'Loading'
const depthModelReady = loadDepthModel()

const depthInputCanvas = document.createElement('canvas')
const depthInputContext = depthInputCanvas.getContext('2d')

function getDepthStats(predictedDepth) {
  const values = predictedDepth.data
  let min = Infinity
  let max = -Infinity

  for (const value of values) {
    min = Math.min(min, value)
    max = Math.max(max, value)
  }

  const height = predictedDepth.dims.at(-2)
  const width = predictedDepth.dims.at(-1)

  return { width, height, min, max }
}

function drawDepthDebugView(predictedDepth, stats) {
  const { width, height, min, max } = stats
  const range = max - min || 1

  if (depthDebugCanvas.width !== width || depthDebugCanvas.height !== height) {
    depthDebugCanvas.width = width
    depthDebugCanvas.height = height
  }

  const imageData = depthDebugContext.createImageData(width, height)

  for (let index = 0; index < predictedDepth.data.length; index += 1) {
    const shade = Math.round(((predictedDepth.data[index] - min) / range) * 255)
    const pixelIndex = index * 4

    imageData.data[pixelIndex] = shade
    imageData.data[pixelIndex + 1] = shade
    imageData.data[pixelIndex + 2] = shade
    imageData.data[pixelIndex + 3] = 255
  }

  depthDebugContext.putImageData(imageData, 0, 0)
}

async function estimateCurrentFrame(depthEstimator) {
  depthInputCanvas.width = video.videoWidth
  depthInputCanvas.height = video.videoHeight
  depthInputContext.drawImage(
    video,
    0,
    0,
    depthInputCanvas.width,
    depthInputCanvas.height,
  )

  return depthEstimator(depthInputCanvas)
}

function waitForNextVideoFrame() {
  return new Promise((resolve) => {
    if ('requestVideoFrameCallback' in video) {
      video.requestVideoFrameCallback(() => resolve())
    } else {
      requestAnimationFrame(() => resolve())
    }
  })
}

async function runDepthLoop(cameraReady) {
  const [cameraStarted, depthEstimator] = await Promise.all([
    cameraReady,
    depthModelReady,
  ])

  if (!cameraStarted || !depthEstimator) return

  let depthFrameRate = 0
  let lastDepthFrameTime = 0
  let hasLoggedDepthStats = false

  while (video.srcObject) {
    await waitForNextVideoFrame()

    try {
      const { predicted_depth: predictedDepth } =
        await estimateCurrentFrame(depthEstimator)
      const stats = getDepthStats(predictedDepth)
      const { width, height, min, max } = stats
      const now = performance.now()
      const instantaneousFrameRate = lastDepthFrameTime
        ? 1000 / (now - lastDepthFrameTime)
        : 0

      lastDepthFrameTime = now
      depthFrameRate = depthFrameRate
        ? depthFrameRate * 0.8 + instantaneousFrameRate * 0.2
        : instantaneousFrameRate

      if (!hasLoggedDepthStats) {
        console.log(
          `Depth output: ${width}x${height}, min=${min.toFixed(4)}, max=${max.toFixed(4)}`,
        )
        hasLoggedDepthStats = true
      }

      drawDepthDebugView(predictedDepth, stats)
      const pointCloudFrame = updatePointCloudGeometry(predictedDepth)
      updatePointCloudColors(
        pointCloudFrame.count,
        pointCloudFrame.width,
        pointCloudFrame.height,
      )

      depthStatus.textContent =
        `Depth ${width}×${height} · ${activeDepthBackend} · ` +
        `${depthFrameRate.toFixed(1)} FPS · D toggles debug`
      depthStatus.dataset.backend = activeDepthBackend
      depthStatus.dataset.frameRate = depthFrameRate.toFixed(1)
    } catch (error) {
      depthStatus.textContent = `Depth inference failed · ${activeDepthBackend}`
      console.error('Unable to estimate depth:', error)
      return
    }
  }
}

function showCameraMessage(message) {
  const cameraMessage = document.createElement('p')
  cameraMessage.className = 'camera-message'
  cameraMessage.textContent = message
  document.body.appendChild(cameraMessage)
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user',
      },
    })

    video.srcObject = stream
    await video.play()

    console.log(`Webcam video size: ${video.videoWidth}x${video.videoHeight}`)
    console.log(
      'Camera intrinsics (60° horizontal FOV):',
      createCameraIntrinsics(video.videoWidth, video.videoHeight),
    )
    ensurePointCloudCapacity(video.videoWidth * video.videoHeight)
    renderWebcamFrame()
    return true
  } catch (error) {
    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
      showCameraMessage('Camera permission was denied.')
    } else if (error.name === 'NotFoundError') {
      showCameraMessage('No camera was found.')
    } else {
      showCameraMessage('Camera access is unavailable.')
      console.error('Unable to start the camera:', error)
    }

    return false
  }
}

const cameraReady = startCamera()
runDepthLoop(cameraReady)

window.addEventListener('keydown', (event) => {
  if (event.key.toLowerCase() !== 'd' || event.repeat) return

  depthDebugCanvas.hidden = !depthDebugCanvas.hidden
  depthStatus.dataset.debugVisible = String(!depthDebugCanvas.hidden)
})

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()

  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(window.innerWidth, window.innerHeight)
})
