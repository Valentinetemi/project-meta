import * as THREE from 'three'
import './style.css'

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

const webcamCanvas = document.createElement('canvas')
webcamCanvas.className = 'webcam-overlay'
webcamCanvas.width = 320
webcamCanvas.height = 240
visionOverlays.appendChild(webcamCanvas)

const webcamContext = webcamCanvas.getContext('2d')

function renderWebcamFrame() {
  webcamContext.drawImage(video, 0, 0, webcamCanvas.width, webcamCanvas.height)

  if ('requestVideoFrameCallback' in video) {
    video.requestVideoFrameCallback(renderWebcamFrame)
  } else {
    requestAnimationFrame(renderWebcamFrame)
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
  } catch (error) {
    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') {
      showCameraMessage('Camera permission was denied.')
    } else if (error.name === 'NotFoundError') {
      showCameraMessage('No camera was found.')
    } else {
      showCameraMessage('Camera access is unavailable.')
      console.error('Unable to start the camera:', error)
    }
  }
}

startCamera()

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()

  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(window.innerWidth, window.innerHeight)
})
