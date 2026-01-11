import React, { useEffect, useRef, useState } from "react"
import { addPropertyControls, ControlType } from "framer"
import * as THREE from "three"
// Post-processing imports removed to prevent loading errors


// --- SHADER SOURCE ---
const vertexShader = `
    attribute vec3 aBarycentric;
    varying vec3 vBarycentric, vNormal, vWorldNormal, vWorldPosition;
    varying vec2 vUv;
    uniform float uTime, uWindSpeed, uAmplitude, uFrequency, uLayerOffset, uChaos, uPointSize;

    float vHash(vec3 p) {
        p = fract(p * 0.1031);
        p += dot(p, p.yzx + 33.33);
        return fract((p.x + p.y) * p.z);
    }

    void main() {
        vUv = uv; vBarycentric = aBarycentric;
        vec3 pos = position;
        float t = uTime * uWindSpeed + uLayerOffset;
        float wave = sin(pos.x * uFrequency * 0.5 + t) * cos(pos.y * uFrequency * 0.3 + t);
        pos.z += (wave + sin((pos.x + pos.y) * uFrequency + t * 1.5) * 0.15) * uAmplitude;
        if(uChaos > 0.0) pos += (vHash(pos + uTime) - 0.5) * uChaos * 2.5;
        
        vec4 worldPosition = modelMatrix * vec4(pos, 1.0);
        vWorldPosition = worldPosition.xyz;
        
        vNormal = normalize(normalMatrix * normal); 
        vWorldNormal = normalize(mat3(modelMatrix) * normal);
        
        gl_PointSize = uPointSize;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    }
`

const fragmentShader = `
    varying vec3 vBarycentric, vNormal, vWorldNormal, vWorldPosition;
    varying vec2 vUv;
    uniform float uTime, uLineWidth, uOpacity, uEnvBrightness, uEnvContrast, uPointSize;
    uniform int uRenderMode, uWireMode;
    uniform vec3 uColor, uColor2, uWireColor, uEmissionColor, uOutlineColor, uCursorLightPos;
    uniform float uEmissionIntensity, uChaos, uCursorLightIntensity, uFireSpeed, uFireScale, uFireIntensity, uTurbSpeed, uTurbScale;
    uniform float uShadowIntensity;
    uniform bool uOutlineEnable, uOutlineEmissive, uUseEnvMap;
    uniform sampler2D uEnvMap2D;
    uniform vec3 uLightPos;

    float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
    float fbm(vec2 p) {
        float v = 0.0; float a = 0.5;
        for (int i = 0; i < 3; ++i) { 
            vec2 i_f = floor(p); vec2 fr = fract(p);
            vec2 u = fr*fr*(3.0-2.0*fr);
            float n = mix(mix(hash(i_f + vec2(0,0)), hash(i_f + vec2(1,0)), u.x),
                       mix(hash(i_f + vec2(0,1)), hash(i_f + vec2(1,1)), u.x), u.y);
            v += a * n; p *= 2.0; a *= 0.5; 
        }
        return v;
    }
    
    // Convert reflection direction to equirectangular UV
    vec2 envMapEquirect(vec3 dir) {
        float phi = atan(dir.z, dir.x);
        float theta = asin(clamp(dir.y, -1.0, 1.0));
        return vec2(phi / 6.28318 + 0.5, theta / 3.14159 + 0.5);
    }

    void main() {
        vec2 uv = vUv;
        if (uRenderMode == 12) {
            float pScale = mix(10.0, 200.0, uLineWidth / 5.0);
            uv = floor(uv * pScale) / pScale;
        }
        vec3 N = normalize(vNormal);
        vec3 V = normalize(cameraPosition - vWorldPosition);
        vec3 R = reflect(-V, normalize(vWorldNormal));
        
        // Sample environment map (2D equirectangular)
        vec3 env = vec3(0.5);
        if (uUseEnvMap) {
            vec2 envUV = envMapEquirect(R);
            env = texture2D(uEnvMap2D, envUV).rgb;
            env = (env * uEnvBrightness - 0.5) * uEnvContrast + 0.5;
        }
        
        vec3 baseCol = uColor;
        if (uRenderMode == 10) { 
            vec2 fireUv = uv * uFireScale; fireUv.y -= uTime * uFireSpeed;
            baseCol = mix(uColor, uColor2, fbm(fireUv) * pow(1.0 - uv.y, 2.0) * uFireIntensity);
        } else if (uRenderMode == 13) { 
            baseCol = mix(uColor, uColor2, fbm(uv * uTurbScale + uTime * uTurbSpeed));
        }
        if (uChaos > 0.0) baseCol += (hash(uv + uTime) - 0.5) * uChaos;

        // Simple custom lighting
        vec3 lightDir = normalize(uLightPos - vWorldPosition);
        float diff = max(dot(N, lightDir), 0.0);
        vec3 ambient = vec3(0.4);
        vec3 addedLights = ambient + vec3(1.0) * diff;

        vec3 curL = normalize(uCursorLightPos - vWorldPosition);
        float curAtten = 1.0 / (1.0 + distance(uCursorLightPos, vWorldPosition) * 0.1);
        addedLights += max(dot(N, curL), 0.0) * uCursorLightIntensity * curAtten;

        vec3 finalColor = baseCol * addedLights;

        if (uRenderMode == 7) { // Chrome
             finalColor = mix(finalColor * 0.1, env, 0.95);
        } 
        else if (uRenderMode == 1) {
            float edge;
            if (uWireMode == 1) edge = vBarycentric.x; else if (uWireMode == 2) edge = vBarycentric.y; else if (uWireMode == 3) edge = vBarycentric.z;
            else edge = min(min(vBarycentric.x, vBarycentric.y), vBarycentric.z);
            finalColor = mix(finalColor, uWireColor, 1.0 - smoothstep(0.0, uLineWidth * 0.05, edge));
        } else if (uRenderMode == 2) {
            float threshold = 1.0 - (uPointSize * 0.02); 
            float maxBary = max(max(vBarycentric.x, vBarycentric.y), vBarycentric.z);
            if (maxBary > threshold) {
                finalColor = uColor;
            } else {
                discard;
            }
        } else if (uRenderMode == 14) {
            float scanline = sin(vWorldPosition.y * 10.0 + uTime * 5.0) * 0.5 + 0.5;
            finalColor = mix(baseCol, vec3(0.0, 1.0, 1.0), scanline * 0.5) + env * 0.5;
        }

        finalColor += uEmissionColor * uEmissionIntensity;
        
        if (uOutlineEnable) {
            float dotNV = max(dot(N, V), 0.0);
            float sil = 1.0 - smoothstep(0.1, 0.15, dotNV);
            vec3 strokeCol = uOutlineColor;
            if (uOutlineEmissive) strokeCol += uEmissionColor * uEmissionIntensity;
            finalColor = mix(finalColor, strokeCol, sil);
        }
        
        gl_FragColor = vec4(finalColor, uOpacity);
    }
`

const STYLE_OPTIONS = ["Solid", "Wireframe", "Points", "Glass", "Reflective", "X-Ray", "Polygon", "Chrome", "Shiny Plastic", "Toon", "Fire", "Rain", "Pixelation", "Turbulence", "Hologram"]
const DISTRIBUTION_OPTIONS = ["Planar Grid", "X-Stack", "Y-Stack", "Z-Stack"]
const VIEW_OPTIONS = ["Manual", "Free", "Front", "Back", "Left", "Right", "Top", "Bottom", "Iso NE", "Iso NW", "Iso SE", "Iso SW"]
const MODE_MAP: Record<string, number> = { "Solid": 0, "Wireframe": 1, "Points": 2, "Glass": 3, "Reflective": 4, "X-Ray": 5, "Polygon": 6, "Chrome": 7, "Shiny Plastic": 8, "Toon": 9, "Fire": 10, "Rain": 11, "Pixelation": 12, "Turbulence": 13, "Hologram": 14 }

const VIEW_PRESETS: Record<string, [number, number, number]> = {
    "Free": [0, 0, 100],
    "Front": [0, 0, 100],
    "Back": [0, 0, -100],
    "Left": [-100, 0, 0],
    "Right": [100, 0, 0],
    "Top": [0, 100, 0.01],
    "Bottom": [0, -100, 0.01],
    "Iso NE": [80, 60, 80],
    "Iso NW": [-80, 60, 80],
    "Iso SE": [80, 60, -80],
    "Iso SW": [-80, 60, -80],
}

const DEFAULT_LAYERS = [
    { enabled: true, style: "Chrome", color: "#6366f1", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
    { enabled: true, style: "Chrome", color: "#ffffff", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
    { enabled: true, style: "Chrome", color: "#6366f1", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
    { enabled: true, style: "Chrome", color: "#ffffff", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
    { enabled: true, style: "Chrome", color: "#6366f1", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
    { enabled: true, style: "Chrome", color: "#ffffff", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
    { enabled: true, style: "Chrome", color: "#6366f1", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
    { enabled: true, style: "Chrome", color: "#ffffff", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
    { enabled: true, style: "Chrome", color: "#6366f1", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
]

const DEFAULT_PROPS = {
    bgColor: "#a5a5a5",
    camera: {
        dragEnabled: true,
        cameraView: "Front",
        zoom: 1.0,
        panX: 0,
        panY: 0,
        camOrbit: 0,
        camElevation: 0,
        camDistance: 100,
    },
    environment: {
        envMapImage: "",
        envBrightness: 1.0,
        envContrast: 1.0,
    },
    lighting: {
        lightX: 15,
        lightY: 25,
        lightZ: 15,
        lightIntensity: 1.0,
        shadowIntensity: 0.3,
    },
    layout: {
        distribution: "Planar Grid",
        spacing: 13.0,
    },
    layers: DEFAULT_LAYERS,
}

export default function ClothStudio(incomingProps: any) {
    // Deep merge props with defaults
    const props = {
        ...DEFAULT_PROPS,
        ...incomingProps,
        camera: { ...DEFAULT_PROPS.camera, ...incomingProps?.camera },
        environment: { ...DEFAULT_PROPS.environment, ...incomingProps?.environment },
        lighting: { ...DEFAULT_PROPS.lighting, ...incomingProps?.lighting },
        layout: { ...DEFAULT_PROPS.layout, ...incomingProps?.layout },
        layers: (incomingProps?.layers && incomingProps.layers.length > 0) ? incomingProps.layers : DEFAULT_LAYERS,
    }

    const { bgColor, camera, environment, lighting, layout, layers } = props
    const { dragEnabled, cameraView, zoom, panX, panY, camOrbit, camElevation, camDistance } = camera
    const { envMapImage, envBrightness, envContrast } = environment
    const { lightX, lightY, lightZ, lightIntensity, shadowIntensity } = lighting
    const { distribution, spacing } = layout

    const containerRef = useRef<HTMLDivElement>(null)
    const engineRef = useRef<any>({ renderer: null, scene: null, camera: null, meshes: [], clock: null, active: false, pivot: null, frameId: 0, lastView: "", envTexture: null, loadedEnvUrl: "" })
    const propsRef = useRef(props)
    propsRef.current = props

    const interactionRef = useRef({
        isOrbiting: false,
        isPanning: false,
        isZooming: false,
        lastX: 0,
        lastY: 0
    })

    useEffect(() => {
        const ctx = engineRef.current
        const envUrl = props.envMapImage

        if (!envUrl || envUrl === ctx.loadedEnvUrl) return

        const loader = new THREE.TextureLoader()
        loader.load(envUrl, (texture: any) => {
            texture.mapping = THREE.EquirectangularReflectionMapping
            ctx.envTexture = texture
            ctx.loadedEnvUrl = envUrl
            ctx.meshes.forEach((mesh: any) => {
                if (mesh?.material?.uniforms) {
                    mesh.material.uniforms.uEnvMap2D.value = texture
                    mesh.material.uniforms.uUseEnvMap.value = true
                }
            })
        })
    }, [props.envMapImage])

    const geometryKey = layers.map((l: any) => l.res).join("-")

    useEffect(() => {
        if (!containerRef.current) return

        const ctx = engineRef.current
        ctx.active = true
        ctx.clock = new THREE.Clock()

        const scene = new THREE.Scene()
        const pivot = new THREE.Group()
        pivot.position.set(0, 0, 0)
        scene.add(pivot)
        ctx.pivot = pivot
        ctx.scene = scene

        const updateSize = () => {
            if (!containerRef.current || !ctx.renderer || !ctx.camera) return
            const width = containerRef.current.clientWidth
            const height = containerRef.current.clientHeight
            const aspect = width / height

            ctx.camera.left = -50 * aspect
            ctx.camera.right = 50 * aspect
            ctx.camera.top = 50
            ctx.camera.bottom = -50
            ctx.camera.updateProjectionMatrix()

            ctx.renderer.setSize(width, height)
            ctx.renderer.setPixelRatio(window.devicePixelRatio)
        }

        const width = containerRef.current.clientWidth
        const height = containerRef.current.clientHeight
        const aspect = width / height
        const camera = new THREE.OrthographicCamera(-50 * aspect, 50 * aspect, 50, -50, -10000, 10000)
        camera.position.set(0, 0, 100)
        camera.lookAt(0, 0, 0)
        ctx.camera = camera

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
        renderer.setSize(width, height)
        renderer.setPixelRatio(window.devicePixelRatio)
        renderer.shadowMap.enabled = true
        renderer.shadowMap.type = THREE.PCFSoftShadowMap
        containerRef.current.appendChild(renderer.domElement)
        ctx.renderer = renderer
        ctx.meshes = []

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4)
        scene.add(ambientLight)

        const dirLight = new THREE.DirectionalLight(0xffffff, 1.0)
        dirLight.position.set(15, 25, 15)
        dirLight.castShadow = true
        dirLight.shadow.mapSize.width = 2048
        dirLight.shadow.mapSize.height = 2048
        dirLight.shadow.camera.near = 0.5
        dirLight.shadow.camera.far = 500
        dirLight.shadow.camera.left = -50
        dirLight.shadow.camera.right = 50
        dirLight.shadow.camera.top = 50
        dirLight.shadow.camera.bottom = -50
        dirLight.shadow.bias = -0.001
        scene.add(dirLight)
        ctx.dirLight = dirLight

        layers.forEach((layerProp: any, i: number) => {
            const res = layerProp.res || 40
            const geo = new THREE.PlaneGeometry(12, 12, res, res).toNonIndexed()
            const bary = new Float32Array(geo.attributes.position.count * 3)
            geo.setAttribute("aBarycentric", new THREE.BufferAttribute(bary, 3))

            for (let j = 0; j < bary.length; j += 9) {
                bary[j] = 1; bary[j + 1] = 0; bary[j + 2] = 0;
                bary[j + 3] = 0; bary[j + 4] = 1; bary[j + 5] = 0;
                bary[j + 6] = 0; bary[j + 7] = 0; bary[j + 8] = 1;
            }

            const uniforms = {
                uTime: { value: 0 }, uWindSpeed: { value: 1.2 }, uAmplitude: { value: 1.0 }, uFrequency: { value: 1.0 },
                uColor: { value: new THREE.Color("#6366f1") }, uColor2: { value: new THREE.Color("#ff0000") },
                uWireColor: { value: new THREE.Color("#000000") }, uLineWidth: { value: 1.0 }, uOpacity: { value: 1.0 },
                uEmissionColor: { value: new THREE.Color("#ffffff") }, uEmissionIntensity: { value: 0 },
                uChaos: { value: 0 }, uPointSize: { value: 10 },
                uCursorLightPos: { value: new THREE.Vector3() }, uCursorLightIntensity: { value: 2.0 },
                uEnvBrightness: { value: 1.0 }, uEnvContrast: { value: 1.0 }, uRenderMode: { value: 7 }, uWireMode: { value: 0 },
                uFireSpeed: { value: 1.5 }, uFireScale: { value: 2.0 }, uFireIntensity: { value: 1.5 },
                uTurbSpeed: { value: 2.0 }, uTurbScale: { value: 10.0 }, uOutlineEnable: { value: false },
                uOutlineColor: { value: new THREE.Color("#000000") }, uOutlineEmissive: { value: false },
                uEnvMap2D: { value: new THREE.Texture() }, uUseEnvMap: { value: false }, uLayerOffset: { value: i * 2.0 },
                uShadowIntensity: { value: 0.3 },
                uLightPos: { value: new THREE.Vector3(15, 25, 15) }
            }

            const mat = new THREE.ShaderMaterial({
                uniforms,
                vertexShader,
                fragmentShader,
                transparent: true,
                side: THREE.DoubleSide
            })
            const mesh = new THREE.Mesh(geo, mat)

            mesh.customDepthMaterial = new THREE.ShaderMaterial({
                vertexShader: vertexShader,
                fragmentShader: THREE.ShaderLib.depth.fragmentShader,
                uniforms: mat.uniforms
            })
            mesh.castShadow = true
            mesh.receiveShadow = true

            pivot.add(mesh)
            ctx.meshes.push(mesh)
        })

        const animate = () => {
            if (!ctx.active) return
            const t = ctx.clock.getElapsedTime()
            const p = propsRef.current

            // Access nested props
            const cam = p.camera || {}
            const env = p.environment || {}
            const light = p.lighting || {}
            const lay = p.layout || {}
            const currentLayers = (p.layers && p.layers.length > 0) ? p.layers : DEFAULT_LAYERS

            ctx.camera.zoom = cam.zoom || 1
            ctx.camera.updateProjectionMatrix()

            const viewName = cam.cameraView || "Front"
            if (viewName === "Manual") {
                const orbit = (cam.camOrbit || 0) * Math.PI / 180
                const elevation = (cam.camElevation || 0) * Math.PI / 180
                const distance = cam.camDistance || 100

                const x = distance * Math.cos(elevation) * Math.sin(orbit)
                const y = distance * Math.sin(elevation)
                const z = distance * Math.cos(elevation) * Math.cos(orbit)

                ctx.camera.position.set(x, y, z)
                ctx.camera.lookAt(0, 0, 0)
                ctx.pivot.rotation.set(0, 0, 0)
            } else if (viewName !== "Free") {
                if (viewName !== ctx.lastView) {
                    const preset = VIEW_PRESETS[viewName]
                    if (preset) {
                        ctx.camera.position.set(preset[0], preset[1], preset[2])
                        ctx.camera.lookAt(0, 0, 0)
                        ctx.pivot.rotation.set(0, 0, 0)

                        const inter = interactionRef.current
                        inter.isOrbiting = false
                        inter.isPanning = false
                        inter.isZooming = false
                    }
                }
            }
            ctx.lastView = viewName

            ctx.pivot.position.x = -(cam.panX || 0)
            ctx.pivot.position.y = (cam.panY || 0)

            const s = lay.spacing || 13
            const dist = lay.distribution || "Planar Grid"

            ctx.meshes.forEach((mesh: any, i: number) => {
                const l = currentLayers[i] || DEFAULT_LAYERS[i]
                mesh.visible = l.enabled !== false
                if (!mesh.visible) return

                if (dist === "Planar Grid") mesh.position.set((i % 3) * s - s, Math.floor(i / 3) * s - s, 0)
                else if (dist === "X-Stack") mesh.position.set(i * s - (4 * s), 0, 0)
                else if (dist === "Y-Stack") mesh.position.set(0, i * s - (4 * s), 0)
                else mesh.position.set(0, 0, i * s - (4 * s))

                const u = mesh.material.uniforms
                u.uTime.value = t
                u.uRenderMode.value = MODE_MAP[l.style] ?? 7
                u.uColor.value.set(l.color || "#6366f1")
                u.uColor2.value.set(l.color2 || "#ff0000")
                u.uEmissionIntensity.value = l.emission || 0
                u.uChaos.value = l.chaos || 0

                if (l.style === "Wireframe") {
                    u.uLineWidth.value = l.wireThickness || 1.0
                    u.uWireColor.value.set(l.wireColor || "#000000")
                } else if (l.style === "Points") {
                    u.uPointSize.value = l.pointSize || 5.0
                } else {
                    u.uLineWidth.value = 1.0
                }

                u.uWindSpeed.value = l.wind || 1.2
                u.uAmplitude.value = l.amp || 1.0
                u.uEnvBrightness.value = env.envBrightness ?? 1.0
                u.uEnvContrast.value = env.envContrast ?? 1.0

                if (ctx.dirLight) {
                    ctx.dirLight.position.set(light.lightX ?? 15, light.lightY ?? 25, light.lightZ ?? 15)
                    ctx.dirLight.intensity = light.lightIntensity ?? 1.0
                }
                u.uShadowIntensity.value = light.shadowIntensity ?? 0.3
                u.uLightPos.value.set(light.lightX ?? 15, light.lightY ?? 25, light.lightZ ?? 15)

                if (l.style === "Fire") {
                    u.uFireSpeed.value = l.wind || 1.5
                    u.uFireScale.value = (l.amp || 1.0) * 2.0
                } else if (l.style === "Turbulence") {
                    u.uTurbSpeed.value = l.wind || 2.0
                    u.uTurbScale.value = (l.amp || 1.0) * 10.0
                }

                mesh.rotation.x += (l.rotX || 0) * 0.01
                mesh.rotation.y += (l.rotY || 0) * 0.01
                mesh.rotation.z += (l.rotZ || 0) * 0.01
            })

            ctx.renderer.render(scene, ctx.camera)
            ctx.frameId = requestAnimationFrame(animate)
        }
        animate()

        let isOrbiting = false
        let isPanning = false
        let isZooming = false
        let lastPos = { x: 0, y: 0 }

        const onDown = (e: MouseEvent) => {
            const camProps = propsRef.current?.camera || {}
            if (!camProps.dragEnabled) return
            lastPos = { x: e.clientX, y: e.clientY }

            if (e.altKey) {
                e.preventDefault()
                if (e.button === 0) {
                    isOrbiting = true
                } else if (e.button === 1) {
                    isPanning = true
                } else if (e.button === 2) {
                    isZooming = true
                }
            } else {
                if (e.button === 0) {
                    isOrbiting = true
                }
            }
        }

        const onMove = (e: MouseEvent) => {
            if (!isOrbiting && !isPanning && !isZooming) return

            const dx = e.clientX - lastPos.x
            const dy = e.clientY - lastPos.y
            lastPos = { x: e.clientX, y: e.clientY }

            if (isOrbiting) {
                pivot.rotation.y += dx * 0.01
                pivot.rotation.x += dy * 0.01
            } else if (isPanning) {
                pivot.position.x -= dx * 0.1
                pivot.position.y += dy * 0.1
            } else if (isZooming) {
                const zoomDelta = 1 + dy * 0.01
                ctx.camera.zoom = Math.max(0.1, Math.min(10, ctx.camera.zoom * zoomDelta))
                ctx.camera.updateProjectionMatrix()
            }
        }

        const onUp = () => {
            isOrbiting = false
            isPanning = false
            isZooming = false
        }

        const onWheel = (e: WheelEvent) => {
            if (!propsRef.current.dragEnabled) return
            e.preventDefault()
            const zoomDelta = e.deltaY > 0 ? 0.9 : 1.1
            ctx.camera.zoom = Math.max(0.1, Math.min(10, ctx.camera.zoom * zoomDelta))
            ctx.camera.updateProjectionMatrix()
        }

        const onContextMenu = (e: MouseEvent) => {
            if (e.altKey) e.preventDefault()
        }

        const container = containerRef.current
        container.addEventListener("mousedown", onDown)
        container.addEventListener("wheel", onWheel, { passive: false })
        container.addEventListener("contextmenu", onContextMenu)
        window.addEventListener("mousemove", onMove)
        window.addEventListener("mouseup", onUp)

        const resizeObserver = new ResizeObserver(() => updateSize())
        resizeObserver.observe(containerRef.current)

        return () => {
            ctx.active = false
            resizeObserver.disconnect()
            cancelAnimationFrame(ctx.frameId)

            container.removeEventListener("mousedown", onDown)
            container.removeEventListener("wheel", onWheel)
            container.removeEventListener("contextmenu", onContextMenu)
            window.removeEventListener("mousemove", onMove)
            window.removeEventListener("mouseup", onUp)

            if (renderer.domElement && containerRef.current) {
                try { containerRef.current.removeChild(renderer.domElement) } catch (e) { }
            }
            renderer.dispose()
        }
    }, [props.zoom, geometryKey])


    const handleMouseDown = (e: React.MouseEvent) => {
        if (!dragEnabled) return
        const inter = interactionRef.current
        inter.lastX = e.clientX
        inter.lastY = e.clientY

        if (e.button === 0) {
            inter.isOrbiting = true
        } else if (e.button === 1) {
            inter.isPanning = true
        } else if (e.button === 2) {
            inter.isZooming = true
        }
    }

    const handleMouseMove = (e: React.MouseEvent) => {
        const inter = interactionRef.current
        const ctx = engineRef.current
        if (!inter.isOrbiting && !inter.isPanning && !inter.isZooming) return
        if (!ctx.pivot) return

        const dx = e.clientX - inter.lastX
        const dy = e.clientY - inter.lastY
        inter.lastX = e.clientX
        inter.lastY = e.clientY

        if (inter.isOrbiting) {
            ctx.pivot.rotation.y += dx * 0.01
            ctx.pivot.rotation.x += dy * 0.01
        } else if (inter.isPanning) {
            ctx.pivot.position.x -= dx * 0.1
            ctx.pivot.position.y += dy * 0.1
        } else if (inter.isZooming) {
            const zoomDelta = 1 + dy * 0.01
            ctx.camera.zoom = Math.max(0.1, Math.min(10, ctx.camera.zoom * zoomDelta))
            ctx.camera.updateProjectionMatrix()
        }
    }

    const handleMouseUp = () => {
        const inter = interactionRef.current
        inter.isOrbiting = false
        inter.isPanning = false
        inter.isZooming = false
    }

    const handleWheel = (e: React.WheelEvent) => {
        if (!dragEnabled) return
        const ctx = engineRef.current
        if (!ctx.camera) return
        const zoomDelta = e.deltaY > 0 ? 0.9 : 1.1
        ctx.camera.zoom = Math.max(0.1, Math.min(10, ctx.camera.zoom * zoomDelta))
        ctx.camera.updateProjectionMatrix()
    }


    return (
        <div
            ref={containerRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onWheel={handleWheel}
            onContextMenu={(e) => e.preventDefault()}
            style={{
                position: "relative",
                width: "100%",
                height: "100%",
                background: bgColor,
                cursor: dragEnabled ? "grab" : "default",
                overflow: "hidden"
            }}
        />
    )
}

// Helper to create layer property controls
const createLayerControls = (num: number, defaultColor: string) => ({
    [`layer${num}Enabled`]: { type: ControlType.Boolean, title: `Layer ${num}`, defaultValue: true },
    [`layer${num}Style`]: { type: ControlType.Enum, title: "Style", options: STYLE_OPTIONS, defaultValue: "Chrome", hidden: (props: any) => !props[`layer${num}Enabled`] },
    [`layer${num}Color`]: { type: ControlType.Color, title: "Color", defaultValue: defaultColor, hidden: (props: any) => !props[`layer${num}Enabled`] },
    [`layer${num}Color2`]: { type: ControlType.Color, title: "Color 2", defaultValue: "#ff0000", hidden: (props: any) => !props[`layer${num}Enabled`] },
    [`layer${num}Emission`]: { type: ControlType.Number, title: "Emission", min: 0, max: 10, step: 0.1, defaultValue: 0, hidden: (props: any) => !props[`layer${num}Enabled`] },
    [`layer${num}Chaos`]: { type: ControlType.Number, title: "Chaos", min: 0, max: 3, step: 0.1, defaultValue: 0, hidden: (props: any) => !props[`layer${num}Enabled`] },
    [`layer${num}Wind`]: { type: ControlType.Number, title: "Speed", min: 0.1, max: 5, step: 0.1, defaultValue: 1.2, hidden: (props: any) => !props[`layer${num}Enabled`] },
    [`layer${num}Amp`]: { type: ControlType.Number, title: "Height", min: 0, max: 10, step: 0.1, defaultValue: 1, hidden: (props: any) => !props[`layer${num}Enabled`] },
    [`layer${num}RotX`]: { type: ControlType.Number, title: "Rotate X", min: -5, max: 5, step: 0.1, defaultValue: 0, hidden: (props: any) => !props[`layer${num}Enabled`] },
    [`layer${num}RotY`]: { type: ControlType.Number, title: "Rotate Y", min: -5, max: 5, step: 0.1, defaultValue: 0, hidden: (props: any) => !props[`layer${num}Enabled`] },
    [`layer${num}RotZ`]: { type: ControlType.Number, title: "Rotate Z", min: -5, max: 5, step: 0.1, defaultValue: 0, hidden: (props: any) => !props[`layer${num}Enabled`] },
    [`layer${num}Res`]: { type: ControlType.Number, title: "Resolution", min: 2, max: 100, step: 1, defaultValue: 40, hidden: (props: any) => !props[`layer${num}Enabled`] },
    [`layer${num}WireThickness`]: { type: ControlType.Number, title: "Wire Thick", min: 0.1, max: 10, step: 0.1, defaultValue: 1.0, hidden: (props: any) => !props[`layer${num}Enabled`] || props[`layer${num}Style`] !== "Wireframe" },
    [`layer${num}WireColor`]: { type: ControlType.Color, title: "Wire Color", defaultValue: "#000000", hidden: (props: any) => !props[`layer${num}Enabled`] || props[`layer${num}Style`] !== "Wireframe" },
    [`layer${num}PointSize`]: { type: ControlType.Number, title: "Point Size", min: 1, max: 20, step: 1, defaultValue: 5, hidden: (props: any) => !props[`layer${num}Enabled`] || props[`layer${num}Style`] !== "Points" },
})

// Framer Property Controls - Organized into collapsible groups
addPropertyControls(ClothStudio, {
    // === APPEARANCE ===
    bgColor: {
        type: ControlType.Color,
        title: "Background",
        defaultValue: "#a5a5a5",
    },

    // === CAMERA GROUP ===
    camera: {
        type: ControlType.Object,
        title: "Camera",
        controls: {
            dragEnabled: {
                type: ControlType.Boolean,
                title: "Interactive",
                defaultValue: true,
            },
            cameraView: {
                type: ControlType.Enum,
                title: "View Preset",
                options: VIEW_OPTIONS,
                defaultValue: "Front",
            },
            zoom: {
                type: ControlType.Number,
                title: "Zoom",
                min: 0.1,
                max: 5,
                step: 0.1,
                defaultValue: 1.0,
            },
            panX: {
                type: ControlType.Number,
                title: "Pan X",
                min: -100,
                max: 100,
                step: 1,
                defaultValue: 0,
            },
            panY: {
                type: ControlType.Number,
                title: "Pan Y",
                min: -100,
                max: 100,
                step: 1,
                defaultValue: 0,
            },
            camOrbit: {
                type: ControlType.Number,
                title: "Heading",
                min: 0,
                max: 360,
                step: 5,
                defaultValue: 0,
            },
            camElevation: {
                type: ControlType.Number,
                title: "Pitch",
                min: -90,
                max: 90,
                step: 5,
                defaultValue: 0,
            },
            camDistance: {
                type: ControlType.Number,
                title: "Position Z",
                min: 10,
                max: 500,
                step: 10,
                defaultValue: 100,
            },
        }
    },

    // === ENVIRONMENT GROUP ===
    environment: {
        type: ControlType.Object,
        title: "Environment",
        controls: {
            envMapImage: {
                type: ControlType.Image,
                title: "HDRI Image",
            },
            envBrightness: {
                type: ControlType.Number,
                title: "Brightness",
                min: 0,
                max: 3,
                step: 0.1,
                defaultValue: 1.0,
            },
            envContrast: {
                type: ControlType.Number,
                title: "Contrast",
                min: 0,
                max: 3,
                step: 0.1,
                defaultValue: 1.0,
            },
        }
    },

    // === LIGHTING GROUP ===
    lighting: {
        type: ControlType.Object,
        title: "Lighting",
        controls: {
            lightX: {
                type: ControlType.Number,
                title: "Light X",
                min: -50,
                max: 50,
                step: 1,
                defaultValue: 15,
            },
            lightY: {
                type: ControlType.Number,
                title: "Light Y",
                min: -50,
                max: 50,
                step: 1,
                defaultValue: 25,
            },
            lightZ: {
                type: ControlType.Number,
                title: "Light Z",
                min: -50,
                max: 50,
                step: 1,
                defaultValue: 15,
            },
            lightIntensity: {
                type: ControlType.Number,
                title: "Intensity",
                min: 0,
                max: 5,
                step: 0.1,
                defaultValue: 1.0,
            },
            shadowIntensity: {
                type: ControlType.Number,
                title: "Shadow",
                min: 0,
                max: 1,
                step: 0.05,
                defaultValue: 0.3,
            },
        }
    },

    // === LAYOUT GROUP ===
    layout: {
        type: ControlType.Object,
        title: "Layout",
        controls: {
            distribution: {
                type: ControlType.Enum,
                title: "Distribution",
                options: DISTRIBUTION_OPTIONS,
                defaultValue: "Planar Grid",
            },
            spacing: {
                type: ControlType.Number,
                title: "Spacing",
                min: 5,
                max: 30,
                step: 1,
                defaultValue: 13,
            },
        }
    },

    // === LAYERS - Using Array for dynamic collapsible items ===
    layers: {
        type: ControlType.Array,
        title: "Cloth Layers",
        maxCount: 9,
        control: {
            type: ControlType.Object,
            controls: {
                enabled: { type: ControlType.Boolean, title: "Enabled", defaultValue: true },
                style: { type: ControlType.Enum, title: "Style", options: STYLE_OPTIONS, defaultValue: "Chrome" },
                color: { type: ControlType.Color, title: "Color", defaultValue: "#6366f1" },
                color2: { type: ControlType.Color, title: "Color 2", defaultValue: "#ff0000" },
                emission: { type: ControlType.Number, title: "Emission", min: 0, max: 10, step: 0.1, defaultValue: 0 },
                chaos: { type: ControlType.Number, title: "Chaos", min: 0, max: 3, step: 0.1, defaultValue: 0 },
                wind: { type: ControlType.Number, title: "Speed", min: 0.1, max: 5, step: 0.1, defaultValue: 1.2 },
                amp: { type: ControlType.Number, title: "Height", min: 0, max: 10, step: 0.1, defaultValue: 1 },
                rotX: { type: ControlType.Number, title: "Rotate X", min: -5, max: 5, step: 0.1, defaultValue: 0 },
                rotY: { type: ControlType.Number, title: "Rotate Y", min: -5, max: 5, step: 0.1, defaultValue: 0 },
                rotZ: { type: ControlType.Number, title: "Rotate Z", min: -5, max: 5, step: 0.1, defaultValue: 0 },
                res: { type: ControlType.Number, title: "Resolution", min: 2, max: 100, step: 1, defaultValue: 40 },
                wireThickness: { type: ControlType.Number, title: "Wire Thickness", min: 0.1, max: 10, step: 0.1, defaultValue: 1.0 },
                wireColor: { type: ControlType.Color, title: "Wire Color", defaultValue: "#000000" },
                pointSize: { type: ControlType.Number, title: "Point Size", min: 1, max: 20, step: 1, defaultValue: 5 },
            }
        },
        defaultValue: [
            { enabled: true, style: "Chrome", color: "#6366f1", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
            { enabled: true, style: "Chrome", color: "#ffffff", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
            { enabled: true, style: "Chrome", color: "#6366f1", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
            { enabled: true, style: "Chrome", color: "#ffffff", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
            { enabled: true, style: "Chrome", color: "#6366f1", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
            { enabled: true, style: "Chrome", color: "#ffffff", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
            { enabled: true, style: "Chrome", color: "#6366f1", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
            { enabled: true, style: "Chrome", color: "#ffffff", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
            { enabled: true, style: "Chrome", color: "#6366f1", color2: "#ff0000", emission: 0, chaos: 0, wind: 1.2, amp: 1, rotX: 0, rotY: 0, rotZ: 0, res: 40, wireThickness: 1, wireColor: "#000000", pointSize: 5 },
        ]
    },
})
