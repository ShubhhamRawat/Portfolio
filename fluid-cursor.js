/**
 * WebGL Fluid Simulation Cursor Effect
 * Scoped exclusively to the Hero section (#home / .hero)
 * Solves 2D Navier-Stokes equations with vorticity confinement & Jacobi pressure iterations.
 */
(function () {
  'use strict';

  var hero = document.querySelector('.hero');
  var canvas = document.querySelector('[data-fluid-cursor]');
  if (!hero || !canvas) return;

  // Performance and Accessibility Guards
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  function isUnsupported() {
    return (
      window.innerWidth < 768 ||
      reduceMotion.matches ||
      !window.matchMedia('(hover: hover) and (pointer: fine)').matches
    );
  }

  if (isUnsupported()) return;

  // Simulation Tuning
  var CFG = {
    SIM_RESOLUTION:       128,  // Grid resolution for velocity calculations
    DYE_RESOLUTION:       512,  // Fine grid for visible smoke ribbons
    DENSITY_DISSIPATION:  2.2,  // Smoke fade rate (higher = disappears faster)
    VELOCITY_DISSIPATION: 1.6,  // Fluid momentum decay
    PRESSURE:             0.8,
    PRESSURE_ITERATIONS:  20,   // Jacobi iterations for incompressible flow
    CURL:                 22,   // Vorticity strength (causes smoke eddies and swirls)
    SPLAT_RADIUS:         0.18, // Cursor footprint size
    SPLAT_FORCE:          5200  // Momentum injected per stroke
  };

  // Palette tuned to Shubham's portfolio (Teal, Amber, Cyan/Emerald)
  var COLORS = [
    [0.10, 0.76, 0.70],  // Luminous Teal (#0f766e accent)
    [0.96, 0.58, 0.15],  // Warm Amber (#d97706 accent)
    [0.20, 0.82, 0.55],  // Emerald glow
    [0.22, 0.65, 0.98]   // Electric Cyan
  ];

  var COLOR_INTENSITY = 0.15;
  var colorIndex = 0;

  // WebGL Context Setup
  var gl = canvas.getContext('webgl2', { alpha: true, depth: false, stencil: false, antialias: false, premultipliedAlpha: false });
  var isWebGL2 = !!gl;

  if (!gl) {
    gl = canvas.getContext('webgl', { alpha: true, depth: false, stencil: false, antialias: false, premultipliedAlpha: false }) ||
         canvas.getContext('experimental-webgl', { alpha: true, depth: false, stencil: false, antialias: false, premultipliedAlpha: false });
  }
  if (!gl) return;

  var halfFloat, supportLinearFiltering;
  if (isWebGL2) {
    gl.getExtension('EXT_color_buffer_float');
    supportLinearFiltering = gl.getExtension('OES_texture_float_linear');
  } else {
    halfFloat = gl.getExtension('OES_texture_half_float');
    supportLinearFiltering = gl.getExtension('OES_texture_half_float_linear');
    if (!halfFloat) return;
  }

  var halfFloatTexType = isWebGL2 ? gl.HALF_FLOAT : halfFloat.HALF_FLOAT_OES;
  var formatRGBA, formatRG, formatR;

  function supportRenderTextureFormat(internalFormat, format, type) {
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, 4, 4, 0, format, type, null);

    var fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    var ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(texture);
    return ok;
  }

  function getSupportedFormat(internalFormat, format, type) {
    if (!supportRenderTextureFormat(internalFormat, format, type)) {
      if (!isWebGL2) return null;
      if (internalFormat === gl.R16F)  return getSupportedFormat(gl.RG16F, gl.RG, type);
      if (internalFormat === gl.RG16F) return getSupportedFormat(gl.RGBA16F, gl.RGBA, type);
      return null;
    }
    return { internalFormat: internalFormat, format: format };
  }

  if (isWebGL2) {
    formatRGBA = getSupportedFormat(gl.RGBA16F, gl.RGBA, halfFloatTexType);
    formatRG   = getSupportedFormat(gl.RG16F,   gl.RG,   halfFloatTexType);
    formatR    = getSupportedFormat(gl.R16F,    gl.RED,  halfFloatTexType);
  } else {
    formatRGBA = getSupportedFormat(gl.RGBA, gl.RGBA, halfFloatTexType);
    formatRG   = formatRGBA;
    formatR    = formatRGBA;
  }

  if (!formatRGBA || !formatRG || !formatR) return;

  function compile(type, source) {
    var shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  function Program(vert, frag) {
    this.program = gl.createProgram();
    gl.attachShader(this.program, vert);
    gl.attachShader(this.program, frag);
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(this.program));
    }
    this.uniforms = {};
    var count = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS);
    for (var i = 0; i < count; i++) {
      var name = gl.getActiveUniform(this.program, i).name;
      this.uniforms[name] = gl.getUniformLocation(this.program, name);
    }
  }
  Program.prototype.bind = function () { gl.useProgram(this.program); };

  var baseVertex = compile(gl.VERTEX_SHADER, [
    'precision highp float;',
    'attribute vec2 aPosition;',
    'varying vec2 vUv;',
    'varying vec2 vL;',
    'varying vec2 vR;',
    'varying vec2 vT;',
    'varying vec2 vB;',
    'uniform vec2 texelSize;',
    'void main () {',
    '  vUv = aPosition * 0.5 + 0.5;',
    '  vL = vUv - vec2(texelSize.x, 0.0);',
    '  vR = vUv + vec2(texelSize.x, 0.0);',
    '  vT = vUv + vec2(0.0, texelSize.y);',
    '  vB = vUv - vec2(0.0, texelSize.y);',
    '  gl_Position = vec4(aPosition, 0.0, 1.0);',
    '}'
  ].join('\n'));

  var copyShader = compile(gl.FRAGMENT_SHADER, [
    'precision mediump float;',
    'precision mediump sampler2D;',
    'varying highp vec2 vUv;',
    'uniform sampler2D uTexture;',
    'void main () { gl_FragColor = texture2D(uTexture, vUv); }'
  ].join('\n'));

  var clearShader = compile(gl.FRAGMENT_SHADER, [
    'precision mediump float;',
    'precision mediump sampler2D;',
    'varying highp vec2 vUv;',
    'uniform sampler2D uTexture;',
    'uniform float value;',
    'void main () { gl_FragColor = value * texture2D(uTexture, vUv); }'
  ].join('\n'));

  var displayShader = compile(gl.FRAGMENT_SHADER, [
    'precision highp float;',
    'precision highp sampler2D;',
    'varying vec2 vUv;',
    'uniform sampler2D uTexture;',
    'void main () {',
    '  gl_FragColor = vec4(texture2D(uTexture, vUv).rgb, 1.0);',
    '}'
  ].join('\n'));

  var splatShader = compile(gl.FRAGMENT_SHADER, [
    'precision highp float;',
    'precision highp sampler2D;',
    'varying vec2 vUv;',
    'uniform sampler2D uTarget;',
    'uniform float aspectRatio;',
    'uniform vec3 color;',
    'uniform vec2 point;',
    'uniform float radius;',
    'void main () {',
    '  vec2 p = vUv - point.xy;',
    '  p.x *= aspectRatio;',
    '  vec3 splat = exp(-dot(p, p) / radius) * color;',
    '  vec3 base = texture2D(uTarget, vUv).xyz;',
    '  gl_FragColor = vec4(base + splat, 1.0);',
    '}'
  ].join('\n'));

  var advectionSource = [
    'precision highp float;',
    'precision highp sampler2D;',
    'varying vec2 vUv;',
    'uniform sampler2D uVelocity;',
    'uniform sampler2D uSource;',
    'uniform vec2 texelSize;',
    'uniform vec2 dyeTexelSize;',
    'uniform float dt;',
    'uniform float dissipation;',
    'vec4 bilerp (sampler2D sam, vec2 uv, vec2 tsize) {',
    '  vec2 st = uv / tsize - 0.5;',
    '  vec2 iuv = floor(st);',
    '  vec2 fuv = fract(st);',
    '  vec4 a = texture2D(sam, (iuv + vec2(0.5, 0.5)) * tsize);',
    '  vec4 b = texture2D(sam, (iuv + vec2(1.5, 0.5)) * tsize);',
    '  vec4 c = texture2D(sam, (iuv + vec2(0.5, 1.5)) * tsize);',
    '  vec4 d = texture2D(sam, (iuv + vec2(1.5, 1.5)) * tsize);',
    '  return mix(mix(a, b, fuv.x), mix(c, d, fuv.x), fuv.y);',
    '}',
    'void main () {',
    '#ifdef MANUAL_FILTERING',
    '  vec2 coord = vUv - dt * bilerp(uVelocity, vUv, texelSize).xy * texelSize;',
    '  vec4 result = bilerp(uSource, coord, dyeTexelSize);',
    '#else',
    '  vec2 coord = vUv - dt * texture2D(uVelocity, vUv).xy * texelSize;',
    '  vec4 result = texture2D(uSource, coord);',
    '#endif',
    '  float decay = 1.0 + dissipation * dt;',
    '  gl_FragColor = result / decay;',
    '}'
  ].join('\n');

  var divergenceShader = compile(gl.FRAGMENT_SHADER, [
    'precision mediump float;',
    'precision mediump sampler2D;',
    'varying highp vec2 vUv;',
    'varying highp vec2 vL;',
    'varying highp vec2 vR;',
    'varying highp vec2 vT;',
    'varying highp vec2 vB;',
    'uniform sampler2D uVelocity;',
    'void main () {',
    '  float L = texture2D(uVelocity, vL).x;',
    '  float R = texture2D(uVelocity, vR).x;',
    '  float T = texture2D(uVelocity, vT).y;',
    '  float B = texture2D(uVelocity, vB).y;',
    '  vec2 C = texture2D(uVelocity, vUv).xy;',
    '  if (vL.x < 0.0)  { L = -C.x; }',
    '  if (vR.x > 1.0)  { R = -C.x; }',
    '  if (vT.y > 1.0)  { T = -C.y; }',
    '  if (vB.y < 0.0)  { B = -C.y; }',
    '  float div = 0.5 * (R - L + T - B);',
    '  gl_FragColor = vec4(div, 0.0, 0.0, 1.0);',
    '}'
  ].join('\n'));

  var curlShader = compile(gl.FRAGMENT_SHADER, [
    'precision mediump float;',
    'precision mediump sampler2D;',
    'varying highp vec2 vUv;',
    'varying highp vec2 vL;',
    'varying highp vec2 vR;',
    'varying highp vec2 vT;',
    'varying highp vec2 vB;',
    'uniform sampler2D uVelocity;',
    'void main () {',
    '  float L = texture2D(uVelocity, vL).y;',
    '  float R = texture2D(uVelocity, vR).y;',
    '  float T = texture2D(uVelocity, vT).x;',
    '  float B = texture2D(uVelocity, vB).x;',
    '  float vorticity = R - L - T + B;',
    '  gl_FragColor = vec4(0.5 * vorticity, 0.0, 0.0, 1.0);',
    '}'
  ].join('\n'));

  var vorticityShader = compile(gl.FRAGMENT_SHADER, [
    'precision highp float;',
    'precision highp sampler2D;',
    'varying vec2 vUv;',
    'varying vec2 vL;',
    'varying vec2 vR;',
    'varying vec2 vT;',
    'varying vec2 vB;',
    'uniform sampler2D uVelocity;',
    'uniform sampler2D uCurl;',
    'uniform float curl;',
    'uniform float dt;',
    'void main () {',
    '  float L = texture2D(uCurl, vL).x;',
    '  float R = texture2D(uCurl, vR).x;',
    '  float T = texture2D(uCurl, vT).x;',
    '  float B = texture2D(uCurl, vB).x;',
    '  float C = texture2D(uCurl, vUv).x;',
    '  vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));',
    '  force /= length(force) + 0.0001;',
    '  force *= curl * C;',
    '  force.y *= -1.0;',
    '  vec2 velocity = texture2D(uVelocity, vUv).xy;',
    '  velocity += force * dt;',
    '  velocity = min(max(velocity, -1000.0), 1000.0);',
    '  gl_FragColor = vec4(velocity, 0.0, 1.0);',
    '}'
  ].join('\n'));

  var pressureShader = compile(gl.FRAGMENT_SHADER, [
    'precision mediump float;',
    'precision mediump sampler2D;',
    'varying highp vec2 vUv;',
    'varying highp vec2 vL;',
    'varying highp vec2 vR;',
    'varying highp vec2 vT;',
    'varying highp vec2 vB;',
    'uniform sampler2D uPressure;',
    'uniform sampler2D uDivergence;',
    'void main () {',
    '  float L = texture2D(uPressure, vL).x;',
    '  float R = texture2D(uPressure, vR).x;',
    '  float T = texture2D(uPressure, vT).x;',
    '  float B = texture2D(uPressure, vB).x;',
    '  float divergence = texture2D(uDivergence, vUv).x;',
    '  float pressure = (L + R + B + T - divergence) * 0.25;',
    '  gl_FragColor = vec4(pressure, 0.0, 0.0, 1.0);',
    '}'
  ].join('\n'));

  var gradientSubtractShader = compile(gl.FRAGMENT_SHADER, [
    'precision mediump float;',
    'precision mediump sampler2D;',
    'varying highp vec2 vUv;',
    'varying highp vec2 vL;',
    'varying highp vec2 vR;',
    'varying highp vec2 vT;',
    'varying highp vec2 vB;',
    'uniform sampler2D uPressure;',
    'uniform sampler2D uVelocity;',
    'void main () {',
    '  float L = texture2D(uPressure, vL).x;',
    '  float R = texture2D(uPressure, vR).x;',
    '  float T = texture2D(uPressure, vT).x;',
    '  float B = texture2D(uPressure, vB).x;',
    '  vec2 velocity = texture2D(uVelocity, vUv).xy;',
    '  velocity.xy -= vec2(R - L, T - B);',
    '  gl_FragColor = vec4(velocity, 0.0, 1.0);',
    '}'
  ].join('\n'));

  var copyProgram            = new Program(baseVertex, copyShader);
  var clearProgram           = new Program(baseVertex, clearShader);
  var splatProgram           = new Program(baseVertex, splatShader);
  var divergenceProgram      = new Program(baseVertex, divergenceShader);
  var curlProgram            = new Program(baseVertex, curlShader);
  var vorticityProgram       = new Program(baseVertex, vorticityShader);
  var pressureProgram        = new Program(baseVertex, pressureShader);
  var gradientSubtractProgram = new Program(baseVertex, gradientSubtractShader);
  var displayProgram         = new Program(baseVertex, displayShader);

  var advectionProgram = new Program(baseVertex, compile(
    gl.FRAGMENT_SHADER,
    (supportLinearFiltering ? '' : '#define MANUAL_FILTERING\n') + advectionSource
  ));

  var quadBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, 1, 1, 1, -1]), gl.STATIC_DRAW);

  var quadIndices = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, quadIndices);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 2, 0, 2, 3]), gl.STATIC_DRAW);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.enableVertexAttribArray(0);

  function blit(target) {
    if (!target) {
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    } else {
      gl.viewport(0, 0, target.width, target.height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    }
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  }

  function createFBO(w, h, internalFormat, format, type, param) {
    gl.activeTexture(gl.TEXTURE0);
    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, param);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, null);

    var fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT);

    return {
      texture: texture,
      fbo: fbo,
      width: w,
      height: h,
      texelSizeX: 1 / w,
      texelSizeY: 1 / h,
      attach: function (id) {
        gl.activeTexture(gl.TEXTURE0 + id);
        gl.bindTexture(gl.TEXTURE_2D, texture);
        return id;
      }
    };
  }

  function createDoubleFBO(w, h, internalFormat, format, type, param) {
    var fbo1 = createFBO(w, h, internalFormat, format, type, param);
    var fbo2 = createFBO(w, h, internalFormat, format, type, param);
    return {
      width: w,
      height: h,
      texelSizeX: fbo1.texelSizeX,
      texelSizeY: fbo1.texelSizeY,
      read: fbo1,
      write: fbo2,
      swap: function () { var t = this.read; this.read = this.write; this.write = t; }
    };
  }

  function resizeFBO(target, w, h, internalFormat, format, type, param) {
    var next = createFBO(w, h, internalFormat, format, type, param);
    copyProgram.bind();
    gl.uniform1i(copyProgram.uniforms.uTexture, target.attach(0));
    blit(next);
    return next;
  }

  function resizeDoubleFBO(target, w, h, internalFormat, format, type, param) {
    if (target.width === w && target.height === h) return target;
    target.read = resizeFBO(target.read, w, h, internalFormat, format, type, param);
    target.write = createFBO(w, h, internalFormat, format, type, param);
    target.width = w;
    target.height = h;
    target.texelSizeX = 1 / w;
    target.texelSizeY = 1 / h;
    return target;
  }

  var dye, velocity, divergence, curl, pressure;
  var filtering = supportLinearFiltering ? gl.LINEAR : gl.NEAREST;

  function getResolution(resolution) {
    var aspect = gl.drawingBufferWidth / gl.drawingBufferHeight;
    if (aspect < 1) aspect = 1 / aspect;
    var min = Math.round(resolution);
    var max = Math.round(resolution * aspect);
    return gl.drawingBufferWidth > gl.drawingBufferHeight
      ? { width: max, height: min }
      : { width: min, height: max };
  }

  function initFramebuffers() {
    var simRes = getResolution(CFG.SIM_RESOLUTION);
    var dyeRes = getResolution(CFG.DYE_RESOLUTION);
    var texType = halfFloatTexType;

    gl.disable(gl.BLEND);

    dye = dye
      ? resizeDoubleFBO(dye, dyeRes.width, dyeRes.height, formatRGBA.internalFormat, formatRGBA.format, texType, filtering)
      : createDoubleFBO(dyeRes.width, dyeRes.height, formatRGBA.internalFormat, formatRGBA.format, texType, filtering);

    velocity = velocity
      ? resizeDoubleFBO(velocity, simRes.width, simRes.height, formatRG.internalFormat, formatRG.format, texType, filtering)
      : createDoubleFBO(simRes.width, simRes.height, formatRG.internalFormat, formatRG.format, texType, filtering);

    divergence = createFBO(simRes.width, simRes.height, formatR.internalFormat, formatR.format, texType, gl.NEAREST);
    curl       = createFBO(simRes.width, simRes.height, formatR.internalFormat, formatR.format, texType, gl.NEAREST);
    pressure   = createDoubleFBO(simRes.width, simRes.height, formatR.internalFormat, formatR.format, texType, gl.NEAREST);
  }

  var heroRect = null;
  function measureHero() {
    heroRect = hero.getBoundingClientRect();
    return heroRect;
  }

  function resizeCanvas() {
    var r = measureHero();
    var dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    var w = Math.max(1, Math.round(r.width * dpr));
    var h = Math.max(1, Math.round(r.height * dpr));
    if (canvas.width === w && canvas.height === h) return false;
    canvas.width = w;
    canvas.height = h;
    return true;
  }

  var pointer = {
    prevX: 0, prevY: 0,
    x: 0, y: 0,
    hasPrev: false,
    color: COLORS[0]
  };

  var splatQueue = [];

  function correctDeltaX(delta) {
    var aspect = heroRect.width / heroRect.height;
    return aspect < 1 ? delta * aspect : delta;
  }

  function correctDeltaY(delta) {
    var aspect = heroRect.width / heroRect.height;
    return aspect > 1 ? delta / aspect : delta;
  }

  function nextColor() {
    var c = COLORS[colorIndex % COLORS.length];
    colorIndex++;
    return [c[0] * COLOR_INTENSITY, c[1] * COLOR_INTENSITY, c[2] * COLOR_INTENSITY];
  }

  // Pointer listener scoped specifically to the hero boundary
  function onPointerMove(e) {
    if (!heroRect) measureHero();

    var x = e.clientX - heroRect.left;
    var y = e.clientY - heroRect.top;

    // Discard any mouse moves outside the hero section (#home)
    if (x < 0 || y < 0 || x > heroRect.width || y > heroRect.height) {
      pointer.hasPrev = false;
      return;
    }

    var nx = x / heroRect.width;
    var ny = 1 - (y / heroRect.height); // WebGL origin is bottom-left

    if (!pointer.hasPrev) {
      pointer.x = pointer.prevX = nx;
      pointer.y = pointer.prevY = ny;
      pointer.hasPrev = true;
      pointer.color = nextColor();
      return;
    }

    pointer.prevX = pointer.x;
    pointer.prevY = pointer.y;
    pointer.x = nx;
    pointer.y = ny;

    var dx = correctDeltaX(nx - pointer.prevX);
    var dy = correctDeltaY(ny - pointer.prevY);
    if (dx === 0 && dy === 0) return;

    splatQueue.push({
      x: nx,
      y: ny,
      dx: dx * CFG.SPLAT_FORCE,
      dy: dy * CFG.SPLAT_FORCE,
      color: pointer.color
    });

    if (splatQueue.length > 12) splatQueue.splice(0, splatQueue.length - 12);
  }

  function correctRadius(radius) {
    var aspect = canvas.width / canvas.height;
    return aspect > 1 ? radius * aspect : radius;
  }

  function splat(x, y, dx, dy, color) {
    gl.disable(gl.BLEND);
    splatProgram.bind();
    gl.uniform1i(splatProgram.uniforms.uTarget, velocity.read.attach(0));
    gl.uniform1f(splatProgram.uniforms.aspectRatio, canvas.width / canvas.height);
    gl.uniform2f(splatProgram.uniforms.point, x, y);
    gl.uniform3f(splatProgram.uniforms.color, dx, dy, 0);
    gl.uniform1f(splatProgram.uniforms.radius, correctRadius(CFG.SPLAT_RADIUS / 100));
    blit(velocity.write);
    velocity.swap();

    gl.uniform1i(splatProgram.uniforms.uTarget, dye.read.attach(0));
    gl.uniform3f(splatProgram.uniforms.color, color[0], color[1], color[2]);
    blit(dye.write);
    dye.swap();
  }

  function step(dt) {
    gl.disable(gl.BLEND);

    // 1. Curl
    curlProgram.bind();
    gl.uniform2f(curlProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(curlProgram.uniforms.uVelocity, velocity.read.attach(0));
    blit(curl);

    // 2. Vorticity Confinement (creates smoke eddies)
    vorticityProgram.bind();
    gl.uniform2f(vorticityProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(vorticityProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(vorticityProgram.uniforms.uCurl, curl.attach(1));
    gl.uniform1f(vorticityProgram.uniforms.curl, CFG.CURL);
    gl.uniform1f(vorticityProgram.uniforms.dt, dt);
    blit(velocity.write);
    velocity.swap();

    // 3. Divergence
    divergenceProgram.bind();
    gl.uniform2f(divergenceProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(divergenceProgram.uniforms.uVelocity, velocity.read.attach(0));
    blit(divergence);

    // 4. Pressure Clear
    clearProgram.bind();
    gl.uniform1i(clearProgram.uniforms.uTexture, pressure.read.attach(0));
    gl.uniform1f(clearProgram.uniforms.value, CFG.PRESSURE);
    blit(pressure.write);
    pressure.swap();

    // 5. Jacobi Iterations for Incompressible Flow
    pressureProgram.bind();
    gl.uniform2f(pressureProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(pressureProgram.uniforms.uDivergence, divergence.attach(0));
    for (var i = 0; i < CFG.PRESSURE_ITERATIONS; i++) {
      gl.uniform1i(pressureProgram.uniforms.uPressure, pressure.read.attach(1));
      blit(pressure.write);
      pressure.swap();
    }

    // 6. Subtract Pressure Gradient
    gradientSubtractProgram.bind();
    gl.uniform2f(gradientSubtractProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    gl.uniform1i(gradientSubtractProgram.uniforms.uPressure, pressure.read.attach(0));
    gl.uniform1i(gradientSubtractProgram.uniforms.uVelocity, velocity.read.attach(1));
    blit(velocity.write);
    velocity.swap();

    // 7. Advect Velocity
    advectionProgram.bind();
    gl.uniform2f(advectionProgram.uniforms.texelSize, velocity.texelSizeX, velocity.texelSizeY);
    if (!supportLinearFiltering) {
      gl.uniform2f(advectionProgram.uniforms.dyeTexelSize, velocity.texelSizeX, velocity.texelSizeY);
    }
    var velocityId = velocity.read.attach(0);
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocityId);
    gl.uniform1i(advectionProgram.uniforms.uSource, velocityId);
    gl.uniform1f(advectionProgram.uniforms.dt, dt);
    gl.uniform1f(advectionProgram.uniforms.dissipation, CFG.VELOCITY_DISSIPATION);
    blit(velocity.write);
    velocity.swap();

    // 8. Advect Dye Smoke
    if (!supportLinearFiltering) {
      gl.uniform2f(advectionProgram.uniforms.dyeTexelSize, dye.texelSizeX, dye.texelSizeY);
    }
    gl.uniform1i(advectionProgram.uniforms.uVelocity, velocity.read.attach(0));
    gl.uniform1i(advectionProgram.uniforms.uSource, dye.read.attach(1));
    gl.uniform1f(advectionProgram.uniforms.dissipation, CFG.DENSITY_DISSIPATION);
    blit(dye.write);
    dye.swap();
  }

  function render() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.disable(gl.BLEND);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    displayProgram.bind();
    gl.uniform1i(displayProgram.uniforms.uTexture, dye.read.attach(0));
    blit(null);
  }

  var lastTime = performance.now();
  var running = false;
  var rafId = 0;
  var ready = false;

  function frame() {
    rafId = 0;
    var now = performance.now();
    var dt = Math.min((now - lastTime) / 1000, 1 / 60);
    lastTime = now;

    if (resizeCanvas()) initFramebuffers();

    while (splatQueue.length) {
      var s = splatQueue.shift();
      splat(s.x, s.y, s.dx, s.dy, s.color);
    }

    step(dt);
    render();

    if (!ready) {
      ready = true;
      canvas.classList.add('is-ready');
    }

    if (running) rafId = requestAnimationFrame(frame);
  }

  function play() {
    if (running) return;
    running = true;
    lastTime = performance.now();
    if (!rafId) rafId = requestAnimationFrame(frame);
  }

  function pause() {
    running = false;
    if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
  }

  // Lifecycle & Event Wiring
  resizeCanvas();
  initFramebuffers();

  // Listen for pointer moves within the window, filtered by hero boundaries
  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('scroll', function () { heroRect = null; }, { passive: true });
  window.addEventListener('resize', function () { heroRect = null; }, { passive: true });

  hero.addEventListener('pointerleave', function () {
    pointer.hasPrev = false;
  });

  // Stop simulation loop entirely when hero is scrolled out of view (0% idle CPU/GPU cost)
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      entries[0].isIntersecting ? play() : pause();
    }, { threshold: 0 }).observe(hero);
  } else {
    play();
  }

  document.addEventListener('visibilitychange', function () {
    document.hidden ? pause() : play();
  });

  play();
})();
