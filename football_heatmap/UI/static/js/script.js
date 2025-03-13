import * as THREE from 'three';
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// Create scene, camera, and renderer
const scene = new THREE.Scene();
const fov = 75;
const aspect = window.innerWidth / window.innerHeight; // Match canvas size
const near = 0.1; // Adjusted near clipping plane
const far = 1000; // Adjusted far clipping plane
const camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
camera.position.set(0, 5, 10); // Adjusted camera position

const renderer = new THREE.WebGLRenderer({ antialias: true }); // Enable antialiasing
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000); // Set background color to black
document.body.appendChild(renderer.domElement);

// OrbitControls for camera interaction
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; // Smooth camera movement
controls.dampingFactor = 0.1; // Lower values make the controls more responsive
controls.enableZoom = true; // Enable zooming
controls.enablePan = true; // Enable panning
controls.minDistance = 1; // Minimum zoom distance
controls.maxDistance = 1000; // Maximum zoom distance
controls.rotateSpeed = 0.5; // Adjust rotation speed
controls.zoomSpeed = 1.0; // Adjust zoom speed
controls.panSpeed = 0.5; // Adjust panning speed

camera.lookAt(scene.position); // Focus on the center

// Add lights
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5); // Soft ambient light
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1); // Strong directional light
directionalLight.position.set(10, 10, 10); // Position the light
scene.add(directionalLight);

const pointLight = new THREE.PointLight(0xffffff, 1, 100); // Point light for additional illumination
pointLight.position.set(0, 10, 0);
pointLight.castShadow = true; // Enable shadow casting
scene.add(pointLight);

// Line connections
const lines = [
    [0, 1], [0, 2], [0, 3], [1, 4], [4, 7], [7, 10], [2, 5], [5, 8], [8, 11],
    [3, 6], [6, 9], [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [16, 18],
    [18, 20], [14, 17], [17, 19], [19, 21]
];
// Function to add an X-Z plane at a specific Y-coordinate
function addXZPlane(minY) {
    const planeSize = 100; // Size of the plane
    const planeGeometry = new THREE.PlaneGeometry(planeSize, planeSize);
    const planeMaterial = new THREE.MeshBasicMaterial({
        color: 0x808080, // Gray color
        side: THREE.DoubleSide, // Render both sides of the plane
        transparent: true,
        opacity: 0.5 // Semi-transparent
    });
    const plane = new THREE.Mesh(planeGeometry, planeMaterial);
    plane.rotation.x = Math.PI / 2; // Rotate the plane to lie flat on the X-Z plane
    plane.position.y = minY; // Position the plane at the specified Y-coordinate
    scene.add(plane);
}

// Function to draw a line between two keypoints
function addLine(p1, p2, color = 0xffffff) {
    const points = [
        new THREE.Vector3(p1.x, p1.y, p1.z),
        new THREE.Vector3(p2.x, p2.y, p2.z)
    ];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color, linewidth: 2 }); // Thicker lines
    const line = new THREE.Line(geometry, material);
    scene.add(line);
}

// Function to create a sphere for keypoints
function addKeypoint(x, y, z) {
    const sphereGeometry = new THREE.SphereGeometry(0.02, 16, 16); // Slightly larger sphere
    const sphereMaterial = new THREE.MeshPhongMaterial({ color: 0xff0000, shininess: 100 }); // Red color with shine
    const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
    sphere.position.set(x, y, z);
    scene.add(sphere);
}

// Function to visualize a set of keypoints
function visualizeKeypoints(keypoints) {
   
 
    keypoints.forEach(({ x, y, z }) => {
        addKeypoint(x, y, z);
    });
    
  
    // Add lines connecting keypoints
    lines.forEach(([startIdx, endIdx]) => {
        if (startIdx < keypoints.length && endIdx < keypoints.length) {
            addLine(keypoints[startIdx], keypoints[endIdx], 0x00ff00); // Green lines
        }
    });
}

// Load keypoints from the server
function loadKeypoints() {
    fetch('/thirdpage/get-kpts-files')
        .then(response => response.json())
        .then(kptsfiles => {
            kptsfiles.forEach((keypoints, index) => {
                
                visualizeKeypoints(keypoints);
            });
        })
        .catch(error => {
            console.error('Error loading keypoints:', error);
        });
}
// Fetch the min_y value from Flask and add the X-Z plane
fetch('/thirdpage/get-min-y')
    .then(response => response.json())
    .then(minY => {
        console.log("Minimum Y value:", minY);
        //addXZPlane(minY); // Add the X-Z plane at the min_y value
    })
    .catch(error => {
        console.error('Error fetching min_y:', error);
    });

// Start loading keypoints
loadKeypoints();

// Animation loop
function animate() {
    requestAnimationFrame(animate);
    controls.update(); // Required if damping is enabled
    renderer.render(scene, camera);
}

animate();

// Handle window resizing
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});