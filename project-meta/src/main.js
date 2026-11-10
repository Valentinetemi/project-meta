import * as THREE from 'three'
import { FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision'
import { pipeline } from '@huggingface/transformers'
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
camera.position.z = 3

const renderer = new THREE.WebGLRenderer({ antialias: true })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.setSize(window.innerWidth, window.innerHeight)
document.querySelector('#app').appendChild(renderer.domElement)

const cube = new THREE.Mesh(
  new THREE.BoxGeometry(1, 1, 1),
  new THREE.MeshNormalMaterial(),
)
scene.add(cube)

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
      const { width, height, min, max } = getDepthStats(predictedDepth)
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

      depthStatus.textContent =
        `Depth ${width}×${height} · ${activeDepthBackend} · ` +
        `${depthFrameRate.toFixed(1)} FPS`
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

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()

  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(window.innerWidth, window.innerHeight)
})
