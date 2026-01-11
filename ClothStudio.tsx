import React, { useEffect, useRef, useState } from "react"
import { addPropertyControls, ControlType } from "framer"
import * as THREE from "three"
// Post-processing imports removed to prevent loading errors


// --- SHADER SOURCE ---
const vertexShader = `
    attribute vec3 aBarycentric;
    varying vec3 vBarycentric, vNormal, vWorldNormal, vWorldPosition;
    varying vec2 vUv;
    varying float vDepth;
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
        
        // Displacement function
        float noise = sin(pos.x * uFrequency * 0.5 + t) * cos(pos.y * uFrequency * 0.3 + t);
        noise += sin((pos.x + pos.z) * uFrequency * 0.8 + t * 0.5) * 0.2;
        float disp = noise * uAmplitude;
        
        // Displace along normal for better 3D simulation
        pos += normal * disp;
        
        if(uChaos > 0.0) pos += (vHash(position + uTime) - 0.5) * uChaos * 2.5;
        
        vec4 worldPosition = modelMatrix * vec4(pos, 1.0);
        vWorldPosition = worldPosition.xyz;
        
        // Improved normal recalculation for arbitrary surfaces
        // We use the original normal and slightly perturbed version 
        // Or for simplicity in this case, we'll use a standard normal derivation
        // for HeightMap displacement generalizing to normal displacement
        
        float eps = 0.01;
        vec3 tangent = normalize(cross(normal, vec3(0.0, 1.0, 0.0)));
        if (length(tangent) < 0.1) tangent = normalize(cross(normal, vec3(1.0, 0.0, 0.0)));
        vec3 bitangent = cross(normal, tangent);
        
        vec3 pT = position + tangent * eps;
        float noiseT = sin(pT.x * uFrequency * 0.5 + t) * cos(pT.y * uFrequency * 0.3 + t);
        noiseT += sin((pT.x + pT.z) * uFrequency * 0.8 + t * 0.5) * 0.2;
        vec3 pT_disp = pT + normal * (noiseT * uAmplitude);
        
        vec3 pB = position + bitangent * eps;
        float noiseB = sin(pB.x * uFrequency * 0.5 + t) * cos(pB.y * uFrequency * 0.3 + t);
        noiseB += sin((pB.x + pB.z) * uFrequency * 0.8 + t * 0.5) * 0.2;
        vec3 pB_disp = pB + normal * (noiseB * uAmplitude);
        
        vec3 modifiedNormal = normalize(cross(pT_disp - pos, pB_disp - pos));
        
        vNormal = normalize(normalMatrix * modifiedNormal);
        vWorldNormal = normalize(mat3(modelMatrix) * modifiedNormal);
        
        vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
        vDepth = -mvPosition.z;
        
        gl_PointSize = uPointSize;
        gl_Position = projectionMatrix * mvPosition;
    }
`

const fragmentShader = `
    varying vec3 vBarycentric, vNormal, vWorldNormal, vWorldPosition;
    varying vec2 vUv;
    varying float vDepth;
    uniform float uTime, uLineWidth, uOpacity, uEnvBrightness, uEnvContrast, uPointSize;
    uniform int uRenderMode, uWireMode;
    uniform vec3 uColor, uColor2, uWireColor, uEmissionColor, uOutlineColor, uCursorLightPos;
    uniform float uEmissionIntensity, uChaos, uCursorLightIntensity;
    uniform float uShadowIntensity, uAOIntensity, uSubsurface;
    uniform float uHeatSpeed, uHeatScale;
    uniform float uSpecular, uRoughness, uMetalness;
    uniform int uOutlineEnable, uOutlineEmissive, uUseEnvMap;
    uniform sampler2D uEnvMap2D;
    uniform vec3 uLightPos, uLightColor;
    uniform int uUseGradient;
    uniform int uGradientDirection;
    uniform int uHeatCustom;
    uniform float uColorAlpha, uColor2Alpha;

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
    
    // Fresnel-Schlick approximation
    float fresnel(float NdotV, float F0) {
        return F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);
    }
    
    // Wrap lighting for soft fabric appearance
    float wrapDiffuse(float NdotL, float wrap) {
        return max(0.0, (NdotL + wrap) / (1.0 + wrap));
    }
    
    // Soft shadow approximation
    float softShadow(vec3 N, vec3 L, float depth) {
        float NdotL = dot(N, L);
        float wrappedDiff = wrapDiffuse(NdotL, 0.5);
        float selfShadow = smoothstep(-0.3, 0.3, NdotL);
        float depthShadow = 1.0 - smoothstep(0.0, 80.0, depth) * 0.4;
        return wrappedDiff * selfShadow * depthShadow;
    }
    
    // Ambient occlusion from curvature
    float calcAO(vec3 N, vec3 V, float aoIntensity) {
        float cavity = 1.0 - abs(dot(N, vec3(0.0, 0.0, 1.0)));
        float ao = mix(1.0, 0.4, cavity * aoIntensity);
        float edgeDark = pow(1.0 - abs(dot(N, V)), 2.0) * 0.3 * aoIntensity;
        return max(ao - edgeDark, 0.15);
    }
    
    // Subsurface scattering for thin fabric
    vec3 subsurfaceScatter(vec3 baseCol, vec3 L, vec3 V, float sssAmount) {
        float backLight = max(0.0, dot(-L, V));
        float scatter = pow(backLight, 3.0) * 0.5;
        return baseCol * 1.3 * scatter * sssAmount;
    }
    
    // Spectral color cycle
    vec3 spectral(float t) {
        vec3 a = vec3(0.5, 0.5, 0.5);
        vec3 b = vec3(0.5, 0.5, 0.5);
        vec3 c = vec3(1.0, 1.0, 1.0);
        vec3 d = vec3(0.0, 0.33, 0.67);
        return a + b * cos(6.28318 * (c * t + d));
    }
    
    // Iridescence helper
    vec3 iridescence(float cosTheta, vec3 base) {
        return base + 0.5 * cos(vec3(0.0, 2.1, 4.2) + 10.0 * cosTheta);
    }
    
    // Heatmap color gradient (blue -> cyan -> green -> yellow -> red)
    vec3 heatmapColor(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c;
        if (t < 0.25) {
            c = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t * 4.0);
        } else if (t < 0.5) {
            c = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.25) * 4.0);
        } else if (t < 0.75) {
            c = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.5) * 4.0);
        } else {
            c = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.75) * 4.0);
        }
        return c;
    }
    
    // Multi-octave turbulence for smooth animation
    float turbulence(vec2 p, float time) {
        float value = 0.0; float amplitude = 1.0; float frequency = 1.0;
        for (int i = 0; i < 4; i++) {
            value += amplitude * abs(fbm(p * frequency + time) * 2.0 - 1.0);
            amplitude *= 0.5; frequency *= 2.0;
        }
        return value;
    }
    
    vec2 hash22(vec2 p) {
        vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
        p3 += dot(p3, p3.yzx + 33.33);
        return fract((p3.xx + p3.yz) * p3.zy);
    }
    
    float voronoi(vec2 x, float time) {
        vec2 n = floor(x);
        vec2 f = fract(x);
        float md = 8.0;
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                vec2 g = vec2(float(i), float(j));
                vec2 o = hash22(n + g);
                o = 0.5 + 0.5 * sin(time + 6.2831 * o);
                vec2 r = g + o - f;
                float d = dot(r, r);
                if (d < md) md = d;
            }
        }
        return sqrt(md);
    }

    // High-Quality Synthetic Studio Environment
    vec3 syntheticStudioEnv(vec3 viewDir, vec3 normal) {
        vec3 R = reflect(-viewDir, normal);
        float horizon = smoothstep(-0.2, 0.2, R.y);
        vec3 skyColor = mix(vec3(0.02, 0.02, 0.05), vec3(0.5, 0.6, 0.8), horizon);
        float softBox1 = pow(max(dot(R, normalize(vec3(1.0, 1.0, 0.5))), 0.0), 64.0) * 5.0;
        float softBox2 = pow(max(dot(R, normalize(vec3(-1.0, 0.5, -1.0))), 0.0), 32.0) * 3.0;
        float softBox3 = pow(max(dot(R, normalize(vec3(0.0, 1.0, -1.0))), 0.0), 128.0) * 8.0;
        float grid = 0.0;
        if (R.y < 0.0) {
            vec2 gridUV = (R.xz / -R.y) * 2.0;
            float gridLine = max(smoothstep(0.9, 0.95, fract(gridUV.x * 5.0)), smoothstep(0.9, 0.95, fract(gridUV.y * 5.0)));
            grid = gridLine * 0.2 * (1.0 - horizon);
        }
        return skyColor + vec3(softBox1 + softBox2 + softBox3) + vec3(grid);
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
        float NdotV = max(dot(N, V), 0.0);
        float finalOpacity = uOpacity;
        
        // Sample environment map with roughness blur
        vec3 env = vec3(0.5);
        if (uUseEnvMap == 1) {
            vec2 envUV = envMapEquirect(R);
            vec2 blur = vec2(uRoughness * 0.02);
            env = texture2D(uEnvMap2D, envUV).rgb;
            env += texture2D(uEnvMap2D, envUV + blur).rgb;
            env += texture2D(uEnvMap2D, envUV - blur).rgb;
            env /= 3.0;
            env = (env * uEnvBrightness - 0.5) * uEnvContrast + 0.5;
        }
        
        
        vec3 baseCol = uColor;
        if (uUseGradient == 1) {
            float gradFactor = vUv.x;
            if (uGradientDirection == 1) gradFactor = vUv.y;
            else if (uGradientDirection == 2) gradFactor = 1.0 - vUv.x;
            else if (uGradientDirection == 3) gradFactor = 1.0 - vUv.y;
            else if (uGradientDirection == 4) gradFactor = length(vUv - 0.5) * 2.0; // Radial
            baseCol = mix(uColor, uColor2, clamp(gradFactor, 0.0, 1.0));
        }
        if (uRenderMode == 10) { 
            vec2 fireUv = uv * uHeatScale; fireUv.y -= uTime * uHeatSpeed;
            float f = fbm(fireUv) * pow(1.0 - uv.y, 2.0) * 1.5;
            baseCol = mix(uColor, uColor2, f);
            finalOpacity *= mix(uColorAlpha, uColor2Alpha, f);
        } else if (uRenderMode == 13) { 
            float turbVal = fbm(uv * uHeatScale + uTime * uHeatSpeed);
            baseCol = mix(uColor, uColor2, turbVal);
            finalOpacity *= mix(uColorAlpha, uColor2Alpha, turbVal);
        }
        if (uChaos > 0.0) baseCol += (hash(uv + uTime) - 0.5) * uChaos;

        // === CLOTH LIGHTING MODEL ===
        vec3 L = normalize(uLightPos - vWorldPosition);
        vec3 H = normalize(L + V);
        float NdotL = max(dot(N, L), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        
        // Ambient occlusion
        float ao = calcAO(N, V, uAOIntensity);
        
        // Soft shadow with self-shadowing and depth
        float shadow = softShadow(N, L, vDepth);
        
        // Diffuse with wrap lighting
        vec3 diffuse = baseCol * shadow * uLightColor;
        
        // Specular (GGX-inspired with roughness)
        float shininess = mix(256.0, 8.0, uRoughness);
        float spec = pow(NdotH, shininess) * NdotL;
        
        // Scale F0 by Specular for dielectrics (0.08 * 0.5 default = 0.04 standard)
        vec3 f0 = mix(vec3(0.08 * uSpecular), baseCol, uMetalness);
        vec3 specColor = f0 * spec;
        
        // Fresnel rim lighting
        float F = fresnel(NdotV, length(f0));
        vec3 fresnelColor = env * F * (1.0 - uRoughness) * 0.5 * uSpecular;
        
        // Subsurface scattering
        vec3 sss = subsurfaceScatter(baseCol, L, V, uSubsurface);
        
        // Environment reflection
        // Attenuate reflection by roughness and specular
        float reflectAtten = (1.0 - uRoughness) * (1.0 - uRoughness); 
        vec3 envReflection = env * mix(0.08 * uSpecular, 1.0, uMetalness) * reflectAtten * ao;
        
        // Ambient
        vec3 ambient = baseCol * 0.12 * ao;
        
        // Combine all lighting
        vec3 finalColor = ambient + diffuse * ao + specColor + fresnelColor + sss + envReflection;
        
        // Apply overall shadow intensity
        finalColor *= mix(1.0, shadow, uShadowIntensity * 0.5);

        // Cursor light with specular
        vec3 curL = normalize(uCursorLightPos - vWorldPosition);
        vec3 curH = normalize(curL + V);
        float curAtten = 1.0 / (1.0 + distance(uCursorLightPos, vWorldPosition) * 0.1);
        float curDiff = wrapDiffuse(dot(N, curL), 0.3);
        float curSpec = pow(max(dot(N, curH), 0.0), shininess);
        finalColor += (baseCol * curDiff * 0.4 + specColor * curSpec * 0.2) * uCursorLightIntensity * curAtten;

        if (uRenderMode == 7) { // Chrome - fully reflective
             finalColor = mix(finalColor * 0.1, env, 0.95);
        } else if (uRenderMode == 4) { // Reflective
             finalColor = mix(finalColor * 0.2, env, 0.7);
        } else if (uRenderMode == 5) { // X-Ray
             float rim = pow(1.0 - NdotV, 3.0);
             finalColor = mix(vec3(0.0), baseCol, rim) + uEmissionColor * rim * 2.0;
             finalOpacity = rim * uOpacity;
        } else if (uRenderMode == 6) { // Polygon / Flat
             vec3 flatN = normalize(cross(dFdx(vWorldPosition), dFdy(vWorldPosition)));
             float flatDiff = max(dot(flatN, L), 0.0);
             finalColor = baseCol * (flatDiff * 0.8 + 0.2);
        } else if (uRenderMode == 8) { // Shiny Plastic
             float specP = pow(NdotH, 128.0) * uSpecular;
             finalColor = mix(baseCol * (NdotL * 0.7 + 0.3), vec3(1.0), specP);
        } else if (uRenderMode == 9) { // Toon
             float toonL = smoothstep(0.4, 0.45, NdotL) * 0.5 + smoothstep(0.7, 0.75, NdotL) * 0.5;
             finalColor = baseCol * mix(0.2, 1.0, toonL);
        } else if (uRenderMode == 3) { // Glass - Ultra High Quality
            float fresnelPower = 4.0;
            float F = pow(1.0 - NdotV, fresnelPower);
            
            float iorR = 1.0 / 1.45;
            float iorG = 1.0 / 1.50;
            float iorB = 1.0 / 1.55;
            
            vec3 refracR = refract(-V, N, iorR);
            vec3 refracG = refract(-V, N, iorG);
            vec3 refracB = refract(-V, N, iorB);
            
            vec3 colR, colG, colB;
            if (uUseEnvMap == 1) {
                colR = texture2D(uEnvMap2D, envMapEquirect(refracR)).r * vec3(1.0, 0.0, 0.0);
                colG = texture2D(uEnvMap2D, envMapEquirect(refracG)).g * vec3(0.0, 1.0, 0.0);
                colB = texture2D(uEnvMap2D, envMapEquirect(refracB)).b * vec3(0.0, 0.0, 1.0);
                colR = (colR * uEnvBrightness - 0.5) * uEnvContrast + 0.5;
                colG = (colG * uEnvBrightness - 0.5) * uEnvContrast + 0.5;
                colB = (colB * uEnvBrightness - 0.5) * uEnvContrast + 0.5;
            } else {
                colR = syntheticStudioEnv(V, normalize(N + refracR * 0.1)).r * vec3(1.0, 0.0, 0.0);
                colG = syntheticStudioEnv(V, normalize(N + refracG * 0.1)).g * vec3(0.0, 1.0, 0.0);
                colB = syntheticStudioEnv(V, normalize(N + refracB * 0.1)).b * vec3(0.0, 0.0, 1.0);
            }
            vec3 envRefract = colR + colG + colB;
            
            vec3 envReflect;
            if (uUseEnvMap == 1) {
                envReflect = texture2D(uEnvMap2D, envMapEquirect(reflect(-V, N))).rgb;
                envReflect = (envReflect * uEnvBrightness - 0.5) * uEnvContrast + 0.5;
            } else {
                envReflect = syntheticStudioEnv(V, N);
            }
            
            vec3 baseRefract = envRefract * baseCol;
            vec3 glassColor = mix(baseRefract, envReflect, F);
            glassColor += specColor * 3.0;
            finalColor = glassColor;
            
            float minAlpha = 0.1;
            float maxAlpha = 0.85;
            finalOpacity = mix(minAlpha, maxAlpha, F) * uOpacity;
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
        } else if (uRenderMode == 11) { // Distance Field
            vec2 grid = uv * vec2(uHeatScale, uHeatScale * 0.2); // Stretched vertical grid
            grid.y -= uTime * uHeatSpeed;
            vec2 id = floor(grid);
            vec2 gv = fract(grid);
            float h = hash(id);
            float h2 = hash(id + 157.31);
            float x = gv.x - (0.2 + 0.6 * h); // Random X position in cell
            float len = 0.2 + 0.8 * h2; // Random length
            float drop = smoothstep(0.1, 0.0, abs(x)) * smoothstep(len, 0.0, abs(gv.y - 0.5));
            baseCol = mix(uColor, uColor2, drop);
            finalOpacity *= mix(uColorAlpha, uColor2Alpha, drop);
        } else if (uRenderMode == 14) {
            float scanline = sin(vWorldPosition.y * uHeatScale + uTime * uHeatSpeed) * 0.5 + 0.5;
            finalColor = mix(baseCol, vec3(0.0, 1.0, 1.0), scanline * 0.5) + env * 0.5;
        } else if (uRenderMode == 15) { // Heat Turbulence
            float turb = turbulence(uv * uHeatScale, uTime * uHeatSpeed);
            turb = turb * 0.5;
            vec3 heatCol = (uHeatCustom == 1) ? mix(uColor, uColor2, turb) : heatmapColor(turb);
            finalColor = heatCol * (max(dot(N, normalize(uLightPos - vWorldPosition)), 0.0) * 0.3 + 0.7) + heatCol * turb * 0.5;
            finalOpacity *= mix(uColorAlpha, uColor2Alpha, turb);
        } else if (uRenderMode == 16) { // Liquid
            float n = fbm(uv * uHeatScale + uTime * uHeatSpeed);
            float n2 = fbm(uv * uHeatScale * 1.5 - uTime * uHeatSpeed * 0.8);
            float liquid = smoothstep(0.4, 0.6, sin(n * 10.0 + n2 * 5.0));
            finalColor = mix(uColor, uColor2, liquid) + env * 0.3;
            finalOpacity *= mix(uColorAlpha, uColor2Alpha, liquid);
        } else if (uRenderMode == 17) { // Voronoi
            float v = voronoi(uv * uHeatScale, uTime * uHeatSpeed);
            finalColor = mix(uColor, uColor2, v) + specColor * (1.0 - v);
            finalOpacity *= mix(uColorAlpha, uColor2Alpha, v);
        } else if (uRenderMode == 18) { // Digital
            vec2 grid = floor(uv * uHeatScale * 5.0);
            float h = hash(grid + floor(uTime * uHeatSpeed * 2.0));
            float mask = smoothstep(0.7, 0.8, h);
            finalColor = mix(vec3(0.0), uColor, mask * (0.5 + 0.5 * sin(uTime * 10.0 + grid.y)));
            finalOpacity = mask * uOpacity * uColorAlpha;
        } else if (uRenderMode == 19) { // Iridescent
            float irid = pow(1.0 - NdotV, 2.0);
            finalColor = iridescence(NdotV, baseCol) + env * irid;
        } else if (uRenderMode == 20) { // Plasma
            vec2 p = -1.0 + 2.0 * uv;
            float mov0 = p.x * p.y + uTime * uHeatSpeed;
            float mov1 = (uv.x + uv.y) * 5.0 + uTime * uHeatSpeed;
            float mov2 = (uv.x - uv.y) * 10.0 + uTime * uHeatSpeed;
            float c1 = sin(mov0 + mov1 + mov2);
            float c2 = cos(mov0 - mov1 + mov2);
            float c3 = sin(mov2);
            finalColor = mix(uColor, uColor2, (c1 + c2 + c3 + 3.0) / 6.0) + env * 0.2;
        } else if (uRenderMode == 21) { // Matrix
            vec2 mUv = uv * uHeatScale * 10.0;
            float row = floor(mUv.y);
            float offset = hash(vec2(floor(mUv.x), row)) * 10.0 + uTime * uHeatSpeed * 5.0;
            float char = hash(vec2(floor(mUv.x), floor(mUv.y + offset)));
            float mask = smoothstep(0.6, 0.8, char);
            finalColor = uColor * mask * (1.0 - fract(mUv.y + offset));
            finalOpacity = mask * uOpacity;
        } else if (uRenderMode == 22) { // Magma
            float noise = turbulence(uv * uHeatScale, uTime * uHeatSpeed);
            float crack = smoothstep(0.4, 0.6, noise);
            finalColor = mix(vec3(0.05, 0.02, 0.01), uColor, crack * crack) + uColor2 * pow(crack, 8.0) * 5.0;
        } else if (uRenderMode == 23) { // Electric
            float n = turbulence(uv * uHeatScale, uTime * uHeatSpeed);
            float beam = smoothstep(0.05, 0.0, abs(n - 0.5));
            finalColor = mix(baseCol * 0.1, uColor, beam) + uColor * beam * 2.0;
            finalOpacity = mix(0.2, 1.0, beam) * uOpacity;
        } else if (uRenderMode == 24) { // Cyber Neon
            float scan = sin(vWorldPosition.y * 20.0 + uTime * 10.0) * 0.5 + 0.5;
            float edge = pow(1.0 - NdotV, 4.0);
            finalColor = baseCol * 0.2 + uColor * edge * 2.0 + uColor * scan * 0.3;
            if (uOutlineEnable == 1) finalColor += uOutlineColor * 2.0;
        } else if (uRenderMode == 25) { // Rainbow Flow
            float flow = uv.x + uv.y + uTime * uHeatSpeed;
            finalColor = spectral(flow) + env * 0.3;
        } else if (uRenderMode == 26) { // Galaxy Pulse
            float stars = pow(hash(uv * 100.0 + floor(uTime * 2.0)), 20.0);
            float neb = fbm(uv * 3.0 + uTime * 0.2);
            finalColor = vec3(stars) + mix(vec3(0.01, 0.0, 0.05), uColor, neb) * 0.5;
            finalOpacity = mix(0.1, 1.0, stars + neb) * uOpacity;
        } else if (uRenderMode == 27) { // Topo Map
            float n = turbulence(uv * uHeatScale, uTime * uHeatSpeed * 0.1);
            float lines = sin(n * 20.0) * 0.5 + 0.5;
            float topo = smoothstep(0.45, 0.5, lines) - smoothstep(0.5, 0.55, lines);
            finalColor = mix(baseCol * 0.2, uColor, topo) + uColor * topo * 2.0;
        } else if (uRenderMode == 28) { // Thin Film
            float n1 = fbm(uv * uHeatScale + uTime * uHeatSpeed * 0.2);
            float n2 = fbm(uv * uHeatScale * 0.5 - uTime * uHeatSpeed * 0.1);
            float thickness = n1 * n2 * 2.0;
            float interference = cos(NdotV * 10.0 + thickness * 20.0);
            vec3 iridCol = spectral(interference * 0.5 + 0.5);
            finalColor = mix(baseCol * 0.5, iridCol, 0.8) + env * pow(1.0 - NdotV, 2.0);
        }

        finalColor += uEmissionColor * uEmissionIntensity;
        if (uOutlineEnable == 1) {
            float dotNV = max(dot(N, V), 0.0);
            float sil = 1.0 - smoothstep(0.1, 0.15, dotNV);
            vec3 strokeCol = uOutlineColor;
            if (uOutlineEmissive == 1) strokeCol += uEmissionColor * uEmissionIntensity;
            finalColor = mix(finalColor, strokeCol, sil);
        }
        
        gl_FragColor = vec4(finalColor, finalOpacity);
    }
`

const STYLE_OPTIONS = ["Solid", "Glass", "Reflective", "X-Ray", "Polygon", "Chrome", "Shiny Plastic", "Toon", "Fire", "Distance Field", "Pixelation", "Turbulence", "Hologram", "Heat Turbulence", "Liquid", "Voronoi", "Digital", "Iridescent", "Plasma", "Matrix", "Magma", "Electric", "Cyber Neon", "Rainbow Flow", "Galaxy Pulse", "Topo Map", "Thin Film"]
const RENDER_TYPES = ["Surface", "Wireframe", "Points"]
const GRADIENT_OPTIONS = ["Horizontal", "Vertical", "Invert Horz", "Invert Vert", "Radial"]
const DISTRIBUTION_OPTIONS = ["Planar Grid", "X-Stack", "Y-Stack", "Z-Stack"]
const VIEW_OPTIONS = ["Manual", "Front", "Back", "Left", "Right", "Top", "Bottom", "Iso NE", "Iso NW", "Iso SE", "Iso SW"]
const MODE_MAP: any = { "Solid": 0, "Wireframe": 1, "Points": 2, "Glass": 3, "Reflective": 4, "X-Ray": 5, "Polygon": 6, "Chrome": 7, "Shiny Plastic": 8, "Toon": 9, "Fire": 10, "Distance Field": 11, "Pixelation": 12, "Turbulence": 13, "Hologram": 14, "Heat Turbulence": 15, "Liquid": 16, "Voronoi": 17, "Digital": 18, "Iridescent": 19, "Plasma": 20, "Matrix": 21, "Magma": 22, "Electric": 23, "Cyber Neon": 24, "Rainbow Flow": 25, "Galaxy Pulse": 26, "Topo Map": 27, "Thin Film": 28 }

const VIEW_PRESETS: any = {
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

const PRIMITIVES = ["Plane", "Sphere", "Box", "Torus", "Cylinder", "Cone", "Icosahedron", "Octahedron", "Dodecahedron", "Tetrahedron", "Torus Knot", "Capsule"]

const DEFAULT_CLOTH_ITEMS = [
    {
        transform: { type: "Sphere", rotX: 0, rotY: 0, rotZ: 0, res: 60 },
        motion: { rotSpeedX: 0, rotSpeedY: 0.1, rotSpeedZ: 0 },
        simulation: { chaos: 0, wind: 1.2, amp: 1, freq: 1 },
        visual: { enabled: true, renderType: "Surface", style: "Iridescent", gradientEnabled: false, gradientDirection: "Horizontal", color: "#6366f1", color2: "#ff0000", emission: 0, wireThickness: 1, wireColor: "#000000", pointSize: 5, specular: 1.5, roughness: 0.1, metalness: 0.0, aoIntensity: 0.6, subsurface: 0.4, heatSpeed: 1.0, heatScale: 5.0, heatCustomColors: false }
    },
    {
        transform: { type: "Torus", rotX: 45, rotY: 0, rotZ: 0, res: 40 },
        motion: { rotSpeedX: 0.05, rotSpeedY: 0.05, rotSpeedZ: 0 },
        simulation: { chaos: 0, wind: 1.5, amp: 2, freq: 2 },
        visual: { enabled: true, renderType: "Surface", style: "Magma", gradientEnabled: false, gradientDirection: "Horizontal", color: "#ff4400", color2: "#ffcc00", emission: 2.0, wireThickness: 1, wireColor: "#000000", pointSize: 5, specular: 0.3, roughness: 0.7, metalness: 0.0, aoIntensity: 0.6, subsurface: 0.4, heatSpeed: 1.0, heatScale: 8.0, heatCustomColors: false }
    },
    {
        transform: { type: "Plane", rotX: 0, rotY: 0, rotZ: 0, res: 40 },
        motion: { rotSpeedX: 0, rotSpeedY: 0, rotSpeedZ: 0 },
        simulation: { chaos: 0.2, wind: 1.2, amp: 1, freq: 1 },
        visual: { enabled: true, renderType: "Surface", style: "Electric", color: "#00ffff", color2: "#ffffff", emission: 5.0, wireThickness: 1, wireColor: "#000000", pointSize: 5, specular: 0.3, roughness: 0.7, metalness: 0.0, aoIntensity: 0.6, subsurface: 0.4, heatSpeed: 2.0, heatScale: 10.0, heatCustomColors: false }
    },
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
        turntableEnabled: false,
        turntableSpeed: 0.5,
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
    clothItems: DEFAULT_CLOTH_ITEMS,
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
        clothItems: (incomingProps?.clothItems && incomingProps.clothItems.length > 0) ? incomingProps.clothItems : DEFAULT_CLOTH_ITEMS,
    };

    const { bgColor, camera, environment, lighting, layout, clothItems } = props
    const { dragEnabled, cameraView, zoom, panX, panY, camOrbit, camElevation, camDistance } = camera
    const { envMapImage, envBrightness, envContrast } = environment
    const { lightX, lightY, lightZ, lightIntensity, shadowIntensity } = lighting
    const { distribution, spacing } = layout

    const containerRef = useRef(null);
    const engineRef = useRef({ renderer: null, scene: null, camera: null, meshes: [], clock: null, active: false, pivot: null, frameId: 0, lastView: "", envTexture: null, loadedEnvUrl: "" });
    const propsRef = useRef<any>(props);
    const mouseRef = useRef(new THREE.Vector2(0, 0))
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
        const envUrl = envMapImage

        if (!envUrl || envUrl === ctx.loadedEnvUrl) return

        const loader = new THREE.TextureLoader()
        loader.load(envUrl, (texture: any) => {
            texture.mapping = THREE.EquirectangularReflectionMapping
            ctx.envTexture = texture
            ctx.loadedEnvUrl = envUrl
            if (ctx.items) {
                ctx.items.forEach((obj: any) => {
                    const m = obj.mesh;
                    const p = obj.points;
                    if (m?.material?.uniforms) {
                        m.material.uniforms.uEnvMap2D.value = texture
                        m.material.uniforms.uUseEnvMap.value = 1
                    }
                    if (p?.material?.uniforms) {
                        p.material.uniforms.uEnvMap2D.value = texture
                        p.material.uniforms.uUseEnvMap.value = 1
                    }
                })
            }
        })
    }, [envMapImage])

    const geometryKey = new Array(9).fill(0).map((_, i) => {
        const item = (clothItems && clothItems[i]) || DEFAULT_CLOTH_ITEMS[i]
        const t = item.transform || DEFAULT_CLOTH_ITEMS[i].transform
        return `${t.type || "Plane"}-${t.res || 40}`
    }).join("-") + `-${clothItems ? clothItems.length : 0}`

    // === SCENE INITIALIZATION (Runs once) ===
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
        ctx.items = []
        ctx.meshes = [] // Keep legacy meshes array for safety/compatibility

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

        // Animation Loop
        const animate = () => {
            if (!ctx.active) return
            const t = ctx.clock.getElapsedTime()
            const p = propsRef.current

            // Access nested props
            const cam = p.camera || {}
            const env = p.environment || {}
            const light = p.lighting || {}
            const lay = p.layout || {}
            const items = p.clothItems || DEFAULT_CLOTH_ITEMS

            // Update global env uniform if needed, but we mostly handle per-mesh
            // Just verifying uUseEnvMap type is set correctly in mesh loops

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
            } else {
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
            if (cam.turntableEnabled && !interactionRef.current.isOrbiting) {
                ctx.pivot.rotation.y += (cam.turntableSpeed || 0.5) * 0.01
            }
            ctx.lastView = viewName

            ctx.pivot.position.x = -(cam.panX || 0)
            ctx.pivot.position.y = (cam.panY || 0)

            const s = lay.spacing || 13
            const dist = lay.distribution || "Planar Grid"

            // Lighting updates
            if (ctx.dirLight) {
                ctx.dirLight.position.set(light.lightX || 15, light.lightY || 25, light.lightZ || 15)
                ctx.dirLight.intensity = light.lightIntensity !== undefined ? light.lightIntensity : 1.0
                // shadowIntensity handled in shader
            }

            // Mesh Updates
            ctx.items.forEach((obj: any, i: number) => {
                const item = items[i] || DEFAULT_CLOTH_ITEMS[i]
                const visL = item.visual || DEFAULT_CLOTH_ITEMS[i].visual
                const simL = item.simulation || DEFAULT_CLOTH_ITEMS[i].simulation
                const transL = item.transform || DEFAULT_CLOTH_ITEMS[i].transform
                const motionL = item.motion || DEFAULT_CLOTH_ITEMS[i].motion
                const l = { ...visL, ...simL, ...transL, ...motionL }
                const rType = l.renderType || "Surface"

                // Helper to update uniforms
                const updateUniforms = (mat: any) => {
                    const uniforms = mat.uniforms
                    uniforms.uTime.value = t
                    uniforms.uWindSpeed.value = l.wind !== undefined ? l.wind : 1.2
                    uniforms.uAmplitude.value = l.amp !== undefined ? l.amp : 1.0
                    uniforms.uFrequency.value = l.freq !== undefined ? l.freq : 1.0
                    uniforms.uChaos.value = l.chaos || 0
                    uniforms.uColor.value.set(l.color || "#6366f1")
                    uniforms.uColor2.value.set(l.color2 || "#ff0000")
                    uniforms.uColorAlpha.value = l.colorAlpha !== undefined ? l.colorAlpha : 1.0
                    uniforms.uColor2Alpha.value = l.color2Alpha !== undefined ? l.color2Alpha : 1.0
                    uniforms.uUseGradient.value = l.gradientEnabled ? 1 : 0
                    const gIdx = l.gradientDirection ? GRADIENT_OPTIONS.indexOf(l.gradientDirection) : 0
                    uniforms.uGradientDirection.value = gIdx !== -1 ? gIdx : 0
                    uniforms.uEmissionIntensity.value = l.emission || 0

                    let rMode = 7
                    if (rType === "Wireframe") rMode = 1
                    else if (rType === "Points") rMode = 2
                    else {
                        const styleVal = MODE_MAP[l.style || "Chrome"]
                        rMode = styleVal !== undefined ? styleVal : 7
                    }

                    uniforms.uRenderMode.value = rMode
                    uniforms.uWireMode.value = 0
                    uniforms.uLineWidth.value = l.wireThickness || 1.0
                    uniforms.uWireColor.value.set(l.wireColor || "#000000")
                    uniforms.uPointSize.value = l.pointSize || 5

                    uniforms.uSpecular.value = l.specular !== undefined ? l.specular : 0.5
                    uniforms.uRoughness.value = l.roughness !== undefined ? l.roughness : 0.3
                    uniforms.uMetalness.value = l.metalness || 0
                    uniforms.uAOIntensity.value = l.aoIntensity !== undefined ? l.aoIntensity : 0.6
                    uniforms.uSubsurface.value = l.subsurface !== undefined ? l.subsurface : 0.4

                    uniforms.uHeatSpeed.value = l.heatSpeed || 1.0
                    uniforms.uHeatScale.value = l.heatScale || 5.0

                    uniforms.uEnvBrightness.value = env.envBrightness || 1.0
                    uniforms.uEnvContrast.value = env.envContrast || 1.0
                    uniforms.uShadowIntensity.value = light.shadowIntensity !== undefined ? light.shadowIntensity : 0.3
                    uniforms.uLightColor.value.set("#ffffff")

                    if (uniforms.uLightPos && ctx.dirLight) {
                        uniforms.uLightPos.value.copy(ctx.dirLight.position)
                    }
                    uniforms.uUseEnvMap.value = ctx.envTexture ? 1 : 0
                    uniforms.uHeatCustom.value = l.heatCustomColors ? 1 : 0
                }

                const { mesh, points } = obj
                if (!l.enabled) {
                    if (mesh) mesh.visible = false
                    if (points) points.visible = false
                    return
                }

                // Base offset for distribution
                let px = 0, py = 0, pz = 0
                if (dist === "Planar Grid") {
                    const row = Math.floor(i / 3)
                    const col = i % 3
                    px = (col - 1) * s; py = (1 - row) * s; pz = 0
                } else if (dist === "X-Stack") {
                    px = i * s - (4 * s); py = 0; pz = 0
                } else if (dist === "Y-Stack") {
                    px = 0; py = i * s - (4 * s); pz = 0
                } else if (dist === "Z-Stack") {
                    px = 0; py = 0; pz = i * s - (4 * s)
                }

                if (mesh && points) {
                    mesh.position.set(px, py, pz)
                    points.position.set(px, py, pz)

                    const rx = (l.rotX || 0) + (l.rotSpeedX || 0) * t * 20
                    const ry = (l.rotY || 0) + (l.rotSpeedY || 0) * t * 20
                    const rz = (l.rotZ || 0) + (l.rotSpeedZ || 0) * t * 20

                    mesh.rotation.x = rx * Math.PI / 180
                    mesh.rotation.y = ry * Math.PI / 180
                    mesh.rotation.z = rz * Math.PI / 180

                    points.rotation.x = rx * Math.PI / 180
                    points.rotation.y = ry * Math.PI / 180
                    points.rotation.z = rz * Math.PI / 180

                    updateUniforms(mesh.material)
                    updateUniforms(points.material)

                    // Visibility switching
                    if (rType === "Wireframe") {
                        mesh.visible = true
                        mesh.material.wireframe = true
                        points.visible = false
                    } else if (rType === "Points") {
                        mesh.visible = false
                        points.visible = true
                    } else {
                        mesh.visible = true
                        mesh.material.wireframe = false
                        points.visible = false
                    }
                }
            })

            if (ctx.renderer && ctx.scene && ctx.camera) {
                // Project mouse to 3D for cursor light
                const vector = new THREE.Vector3(mouseRef.current.x, mouseRef.current.y, 0.5)
                vector.unproject(ctx.camera)
                const dir = vector.sub(ctx.camera.position).normalize()
                const dist = -ctx.camera.position.z / dir.z
                const pos = ctx.camera.position.clone().add(dir.multiplyScalar(dist))

                if (ctx.items) {
                    ctx.items.forEach((obj: any) => {
                        const { mesh, points } = obj
                        if (mesh && mesh.material.uniforms.uCursorLightPos) {
                            mesh.material.uniforms.uCursorLightPos.value.copy(pos)
                            mesh.material.uniforms.uTime.value = t
                            if (points) {
                                points.material.uniforms.uCursorLightPos.value.copy(pos)
                                points.material.uniforms.uTime.value = t
                            }
                        }
                    })
                }

                ctx.renderer.render(ctx.scene, ctx.camera);
            }
            ctx.frameId = requestAnimationFrame(animate);
        }

        // Start Loop
        animate()

        // Event Listeners - These are now handled by React event handlers on the container div
        // and the interactionRef state.
        // The manual window/container event listeners are removed to prevent conflicts
        // and ensure proper cleanup.

        const resizeObserver = new ResizeObserver(() => updateSize())
        resizeObserver.observe(containerRef.current)

        return () => {
            ctx.active = false
            if (ctx.frameId) cancelAnimationFrame(ctx.frameId)
            resizeObserver.disconnect()

            if (renderer.domElement && containerRef.current) {
                try { containerRef.current.removeChild(renderer.domElement) } catch (e) { }
            }
            renderer.dispose()
        }
    }, [])

    // === MESH RECREATION (Runs on geometryKey change) ===
    useEffect(() => {
        const ctx = engineRef.current
        if (!ctx.scene || !ctx.pivot) return

        // Clear existing items
        // Clear existing items
        if (ctx.items) {
            ctx.items.forEach((obj: any) => {
                if (obj.mesh) {
                    if (obj.mesh.geometry) obj.mesh.geometry.dispose();
                    ctx.pivot.remove(obj.mesh);
                }
                if (obj.points) {
                    if (obj.points.geometry) obj.points.geometry.dispose();
                    ctx.pivot.remove(obj.points);
                }
            });
        }
        ctx.items = []

        // Create new meshes
        const itemsToRender = Array.isArray(clothItems) ? clothItems : DEFAULT_CLOTH_ITEMS
        itemsToRender.forEach((item: any, i: number) => {
            const transL = item.transform || DEFAULT_CLOTH_ITEMS[i].transform
            const res = transL.res || 40

            const type = transL.type || "Plane"
            let geo: any
            switch (type) {
                case "Sphere": geo = new THREE.SphereGeometry(6, res * 2, res); break;
                case "Box": geo = new THREE.BoxGeometry(10, 10, 10, res, res, res); break;
                case "Torus": geo = new THREE.TorusGeometry(5, 2, res, res * 2); break;
                case "Cylinder": geo = new THREE.CylinderGeometry(5, 5, 12, res, res); break;
                case "Cone": geo = new THREE.ConeGeometry(6, 12, res, res); break;
                case "Icosahedron": geo = new THREE.IcosahedronGeometry(8, Math.floor(res / 8)); break;
                case "Octahedron": geo = new THREE.OctahedronGeometry(8, Math.floor(res / 8)); break;
                case "Dodecahedron": geo = new THREE.DodecahedronGeometry(8, Math.floor(res / 8)); break;
                case "Tetrahedron": geo = new THREE.TetrahedronGeometry(8, Math.floor(res / 8)); break;
                case "Torus Knot": geo = new THREE.TorusKnotGeometry(5, 1.5, res * 4, res); break;
                case "Capsule": geo = new THREE.CapsuleGeometry(4, 8, Math.floor(res / 2), res); break;
                default: geo = new THREE.PlaneGeometry(12, 12, res, res);
            }
            geo = geo.toNonIndexed()
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
                uOutlineEnable: { value: 0 },
                uOutlineColor: { value: new THREE.Color("#000000") }, uOutlineEmissive: { value: 0 },
                uEnvMap2D: { value: new THREE.Texture() }, uUseEnvMap: { value: 0 }, uLayerOffset: { value: i * 2.0 },
                uShadowIntensity: { value: 0.3 },
                uLightPos: { value: new THREE.Vector3(15, 25, 15) },
                uSpecular: { value: 0.5 },
                uRoughness: { value: 0.5 },
                uMetalness: { value: 0.0 },
                uAOIntensity: { value: 0.5 },
                uSubsurface: { value: 0.3 },
                uLightColor: { value: new THREE.Color("#ffffff") },
                uHeatSpeed: { value: 1.0 },
                uHeatScale: { value: 5.0 },
                uHeatCustom: { value: 0 },
                uColorAlpha: { value: 1.0 },
                uColor2Alpha: { value: 1.0 },
                uUseGradient: { value: 0 },
                uGradientDirection: { value: 0 }
            }

            const mat = new THREE.ShaderMaterial({
                uniforms,
                vertexShader,
                fragmentShader,
                transparent: true,
                side: THREE.DoubleSide
            })
            const mesh = new THREE.Mesh(geo, mat)
            const points = new THREE.Points(geo, mat.clone())

            ctx.pivot.add(mesh)
            ctx.pivot.add(points)

            mesh.visible = false
            points.visible = false

            mesh.customDepthMaterial = new THREE.ShaderMaterial({
                vertexShader: vertexShader,
                fragmentShader: THREE.ShaderLib.depth.fragmentShader,
                uniforms: mat.uniforms
            })
            mesh.castShadow = true
            mesh.receiveShadow = true

            if (ctx.envTexture) {
                mat.uniforms.uEnvMap2D.value = ctx.envTexture
                mat.uniforms.uUseEnvMap.value = 1
            }

            ctx.items.push({ mesh, points, geo })
        })

    }, [geometryKey])

    // === INTERACTION HANDLERS ===
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

        // Always update mouse coordinates for the cursor light
        if (containerRef.current) {
            const rect = containerRef.current.getBoundingClientRect()
            mouseRef.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
            mouseRef.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
        }

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
    }

    return (
        <div
            ref={containerRef}
            style={{
                width: "100%",
                height: "100%",
                backgroundColor: bgColor,
                overflow: "hidden",
                position: "relative",
                cursor: dragEnabled ? "grab" : "default"
            }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onContextMenu={function (e) { e.preventDefault(); }}
        />
    );
}


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
            turntableEnabled: {
                type: ControlType.Boolean,
                title: "Turntable",
                defaultValue: false,
            },
            turntableSpeed: {
                type: ControlType.Number,
                title: "Table Speed",
                min: -2,
                max: 2,
                step: 0.1,
                defaultValue: 0.5,
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

    // === CLOTH ITEMS ===
    clothItems: {
        type: ControlType.Array,
        title: "Cloth Items",
        itemTitle: "itemName",
        maxCount: 9,
        control: {
            type: ControlType.Object,
            controls: {
                itemName: { type: ControlType.String, defaultValue: "Cloth", hidden: () => true },
                transform: {
                    type: ControlType.Object,
                    title: "Transform",
                    controls: {
                        type: { type: ControlType.Enum, title: "Primitive", options: PRIMITIVES, defaultValue: "Plane" },
                        rotX: { type: ControlType.Number, title: "Rotate X", min: -180, max: 180, step: 1, defaultValue: 0 },
                        rotY: { type: ControlType.Number, title: "Rotate Y", min: -180, max: 180, step: 1, defaultValue: 0 },
                        rotZ: { type: ControlType.Number, title: "Rotate Z", min: -180, max: 180, step: 1, defaultValue: 0 },
                        res: { type: ControlType.Number, title: "Polycount", min: 2, max: 100, step: 1, defaultValue: 40 },
                    }
                },
                motion: {
                    type: ControlType.Object,
                    title: "Motion",
                    controls: {
                        rotSpeedX: { type: ControlType.Number, title: "Speed X", min: -5, max: 5, step: 0.1, defaultValue: 0 },
                        rotSpeedY: { type: ControlType.Number, title: "Speed Y (Turntable)", min: -5, max: 5, step: 0.1, defaultValue: 0 },
                        rotSpeedZ: { type: ControlType.Number, title: "Speed Z", min: -5, max: 5, step: 0.1, defaultValue: 0 },
                    }
                },
                simulation: {
                    type: ControlType.Object,
                    title: "Simulation",
                    controls: {
                        chaos: { type: ControlType.Number, title: "Chaos", min: 0, max: 3, step: 0.1, defaultValue: 0 },
                        wind: { type: ControlType.Number, title: "Wind Speed", min: 0.1, max: 5, step: 0.1, defaultValue: 1.2 },
                        amp: { type: ControlType.Number, title: "Amplitude", min: 0, max: 10, step: 0.1, defaultValue: 1 },
                        freq: { type: ControlType.Number, title: "Frequency", min: 0, max: 10, step: 0.1, defaultValue: 1 },
                    }
                },
                visual: {
                    type: ControlType.Object,
                    title: "Visual",
                    controls: {
                        enabled: { type: ControlType.Boolean, title: "Enabled", defaultValue: true },
                        renderType: { type: ControlType.Enum, title: "Render Mode", options: RENDER_TYPES, defaultValue: "Surface" },
                        style: { type: ControlType.Enum, title: "Style", options: STYLE_OPTIONS, defaultValue: "Solid", hidden: function (props) { return props.renderType !== "Surface" } },
                        heatSpeed: { type: ControlType.Number, title: "Speed", min: 0.1, max: 10, step: 0.1, defaultValue: 1.0, hidden: function (props) { return props.renderType !== "Surface" || (props.style === "Solid" || props.style === "Glass" || props.style === "Reflective" || props.style === "X-Ray" || props.style === "Polygon" || props.style === "Chrome" || props.style === "Shiny Plastic" || props.style === "Toon" || props.style === "Iridescent") } },
                        heatScale: { type: ControlType.Number, title: "Zoom", min: 0.1, max: 50, step: 0.5, defaultValue: 5.0, hidden: function (props) { return props.renderType !== "Surface" || (props.style === "Solid" || props.style === "Glass" || props.style === "Reflective" || props.style === "X-Ray" || props.style === "Polygon" || props.style === "Chrome" || props.style === "Shiny Plastic" || props.style === "Toon" || props.style === "Iridescent" || props.style === "Rainbow Flow") } },
                        heatCustomColors: { type: ControlType.Boolean, title: "Custom Colors", defaultValue: false, hidden: function (props) { return props.style !== "Heat Turbulence" } },
                        gradientEnabled: { type: ControlType.Boolean, title: "Use Gradient", defaultValue: false, hidden: function (props) { return props.renderType !== "Surface" || (props.style !== "Solid" && props.style !== "Glass" && props.style !== "Reflective" && props.style !== "X-Ray" && props.style !== "Polygon" && props.style !== "Chrome" && props.style !== "Shiny Plastic" && props.style !== "Toon" && props.style !== "Iridescent" && props.style !== "Cyber Neon") } },
                        gradientDirection: { type: ControlType.Enum, title: "Gradient Dir", options: GRADIENT_OPTIONS, defaultValue: "Horizontal", hidden: function (props) { return props.renderType !== "Surface" || !props.gradientEnabled || (props.style !== "Solid" && props.style !== "Glass" && props.style !== "Reflective" && props.style !== "X-Ray" && props.style !== "Polygon" && props.style !== "Chrome" && props.style !== "Shiny Plastic" && props.style !== "Toon" && props.style !== "Iridescent" && props.style !== "Cyber Neon") } },
                        color: { type: ControlType.Color, title: "Color A", defaultValue: "#6366f1", hidden: function (props) { return props.renderType === "Wireframe" } },
                        colorAlpha: { type: ControlType.Number, title: "Color A Opacity", min: 0, max: 1, step: 0.05, defaultValue: 1.0, hidden: function (props) { return props.renderType !== "Surface" || (props.style === "Solid" || props.style === "Glass" || props.style === "Reflective" || props.style === "Polygon" || props.style === "Chrome" || props.style === "Shiny Plastic" || props.style === "Toon" || props.style === "Iridescent") } },
                        color2: { type: ControlType.Color, title: "Color B", defaultValue: "#ff0000", hidden: function (props) { return props.renderType !== "Surface" || (!props.gradientEnabled && props.style !== "Fire" && props.style !== "Distance Field" && props.style !== "Turbulence" && props.style !== "Heat Turbulence" && props.style !== "Liquid" && props.style !== "Voronoi" && props.style !== "Digital" && props.style !== "Plasma" && props.style !== "Matrix" && props.style !== "Magma" && props.style !== "Electric" && props.style !== "Cyber Neon" && props.style !== "Thin Film") } },
                        color2Alpha: { type: ControlType.Number, title: "Color B Opacity", min: 0, max: 1, step: 0.05, defaultValue: 1.0, hidden: function (props) { return props.renderType !== "Surface" || (props.style === "Solid" || props.style === "Glass" || props.style === "Reflective" || props.style === "Polygon" || props.style === "Chrome" || props.style === "Shiny Plastic" || props.style === "Toon" || props.style === "Iridescent") } },
                        specular: { type: ControlType.Number, title: "Specular", min: 0, max: 2, step: 0.05, defaultValue: 0.5, hidden: function (props) { return props.renderType !== "Surface" } },
                        roughness: { type: ControlType.Number, title: "Roughness", min: 0, max: 1, step: 0.05, defaultValue: 0.3, hidden: function (props) { return props.renderType !== "Surface" } },
                        metalness: { type: ControlType.Number, title: "Metalness", min: 0, max: 1, step: 0.05, defaultValue: 0.0, hidden: function (props) { return props.renderType !== "Surface" } },
                        aoIntensity: { type: ControlType.Number, title: "AO Intensity", min: 0, max: 1, step: 0.05, defaultValue: 0.6, hidden: function (props) { return props.renderType !== "Surface" } },
                        subsurface: { type: ControlType.Number, title: "Subsurface", min: 0, max: 1, step: 0.05, defaultValue: 0.4, hidden: function (props) { return props.renderType !== "Surface" } },
                        emission: { type: ControlType.Number, title: "Emission", min: 0, max: 10, step: 0.1, defaultValue: 0, hidden: function (props) { return props.renderType !== "Surface" } },
                        wireThickness: { type: ControlType.Number, title: "Size / Thick", min: 0.1, max: 10, step: 0.1, defaultValue: 1.0, hidden: function (props) { return props.renderType !== "Wireframe" && props.style !== "Pixelation" } },
                        wireColor: { type: ControlType.Color, title: "Wire Color", defaultValue: "#000000", hidden: function (props) { return props.renderType !== "Wireframe" } },
                        pointSize: { type: ControlType.Number, title: "Point Size", min: 1, max: 50, step: 1, defaultValue: 5, hidden: function (props) { return props.renderType !== "Points" } },
                    }
                }
            }
        },
        defaultValue: DEFAULT_CLOTH_ITEMS
    },
})
