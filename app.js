// ============================================================
//  YOLO11 REALTIME WEB APP (OPTIMIZED + COLOR + ANTI-JITTER)
// YOLO11n
// Input: 320 x 320
// Output: [1, 18, 8400]
// ============================================================


// ------------------------------------------------------------
// CONFIGURATION
// ------------------------------------------------------------

const MODEL_PATH = "./best.onnx";

const INPUT_SIZE = 320;

const CONF_THRESHOLD = 0.6;

const IOU_THRESHOLD = 0.5;


// ============================================================
// CLASS NAMES (14 CLASSES)
// ============================================================

const CLASS_NAMES = [
    "chieu_tim",
    "chuoi_duoc_do_rung",
    "chuoi_mo_ket",
    "dam_but_cam",
    "dam_but_vang",
    "hong",
    "huynh_anh",
    "mao_ga_duoi_cong",
    "mau_don_do",
    "mau_don_trang",
    "mau_don_vang",
    "mia_do",
    "mo_trang",
    "ngoc_but"
];


// ============================================================
// CLASS COLORS (MÀU RIÊNG CHO TỪNG CLASS)
// ============================================================

const CLASS_COLORS = [
    "#9400D3", // 0: chieu_tim (Tím)
    "#DC143C", // 1: chuoi_duoc_do_rung (Đỏ sẫm)
    "#FF4500", // 2: chuoi_mo_ket (Đỏ cam)
    "#FF8C00", // 3: dam_but_cam (Cam)
    "#FFD700", // 4: dam_but_vang (Vàng)
    "#FF69B4", // 5: hong (Hồng)
    "#FFFF00", // 6: huynh_anh (Vàng sáng)
    "#C71585", // 7: mao_ga_duoi_cong (Đỏ hồng)
    "#FF0000", // 8: mau_don_do (Đỏ tươi)
    "#00CED1", // 9: mau_don_trang (Xanh ngọc)
    "#DAA520", // 10: mau_don_vang (Vàng đồng)
    "#800000", // 11: mia_do (Đỏ bầm)
    "#B0E0E6", // 12: mo_trang (Xanh nhạt)
    "#32CD32"  // 13: ngoc_but (Xanh lá)
];


// ============================================================
// GLOBAL VARIABLES & BUFFERS
// ============================================================

let session = null;

let stream = null;

let running = false;

let lastTime = performance.now();

let frameCounter = 0;

let fps = 0;

const CHANNEL_SIZE = INPUT_SIZE * INPUT_SIZE;
const PRE_ALLOCATED_FLOAT_DATA = new Float32Array(3 * CHANNEL_SIZE);

// ============================================================
// ANTI-JITTER TRACKING SYSTEM (LÀM MƯỢT KHUNG HÌNH)
// ============================================================
let previousDetections = [];
const SMOOTHING_ALPHA = 0.5; // Hệ số làm mượt (0.1: rất mượt/chậm, 0.9: nhạy/dễ rung). Giá trị 0.5 là tối ưu.
const MAX_MISSING_FRAMES = 5;  // Số khung hình giữ lại box cũ nếu model bị mất phát hiện chớp nhoáng


// ============================================================
// DOM
// ============================================================

const video =
    document.getElementById("video");

const overlay =
    document.getElementById("overlay");

const overlayCtx =
    overlay.getContext("2d");

const processingCanvas =
    document.getElementById("processingCanvas");

const processingCtx =
    processingCanvas.getContext("2d", {
        willReadFrequently: true
    });

const statusElement =
    document.getElementById("status");

const fpsElement =
    document.getElementById("fps");

const objectsElement =
    document.getElementById("objects");

const detectionsElement =
    document.getElementById("detections");

const loadButton =
    document.getElementById("loadButton");

const cameraButton =
    document.getElementById("cameraButton");

const stopButton =
    document.getElementById("stopButton");


// ============================================================
// STATUS
// ============================================================

function setStatus(text) {

    statusElement.textContent = text;

}


// ============================================================
// LOAD MODEL
// ============================================================

async function loadModel() {

    try {

        setStatus("Loading YOLO model...");

        loadButton.disabled = true;

        if (typeof ort !== 'undefined' && ort.env && ort.env.wasm) {
            ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4);
            ort.env.wasm.simd = true;
        }

        try {

            session =
                await ort.InferenceSession.create(
                    MODEL_PATH,
                    {
                        executionProviders: [
                            "webgl"
                        ]
                    }
                );

            setStatus("YOLO loaded - WebGL");

        } catch (webglError) {

            console.warn(
                "WebGL failed:",
                webglError
            );

            session =
                await ort.InferenceSession.create(
                    MODEL_PATH,
                    {
                        executionProviders: [
                            "wasm"
                        ]
                    }
                );

            setStatus("YOLO loaded - WASM");

        }

        console.log(
            "Input names:",
            session.inputNames
        );

        console.log(
            "Output names:",
            session.outputNames
        );

    } catch (error) {

        console.error(error);

        setStatus(
            "Model loading failed"
        );

        alert(
            "Cannot load best.onnx.\n\n" +
            "Check that best.onnx is in the same folder as index.html."
        );

        loadButton.disabled = false;

    }

}


// ============================================================
// CAMERA
// ============================================================

async function startCamera() {

    if (!session) {

        alert(
            "Load the model first."
        );

        return;

    }

    try {

        setStatus(
            "Requesting camera..."
        );

        stream =
            await navigator.mediaDevices.getUserMedia(
                {
                    video: {
                        facingMode: {
                            ideal: "environment"
                        },

                        width: {
                            ideal: 640
                        },

                        height: {
                            ideal: 480
                        }
                    },

                    audio: false
                }
            );

        video.srcObject = stream;

        await video.play();

        setupOverlay();

        running = true;

        setStatus(
            "Camera running"
        );

        requestAnimationFrame(
            inferenceLoop
        );

    } catch (error) {

        console.error(error);

        setStatus(
            "Camera error"
        );

        alert(
            "Camera access failed.\n\n" +
            "Make sure you opened the website using HTTPS and allowed Camera permission."
        );

    }

}


// ============================================================
// STOP CAMERA
// ============================================================

function stopCamera() {

    running = false;

    if (stream) {

        stream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        stream = null;

    }

    video.srcObject = null;

    overlayCtx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    objectsElement.textContent = "0";

    detectionsElement.textContent =
        "No detection";

    previousDetections = [];

    setStatus(
        session
            ? "Model loaded"
            : "Model not loaded"
    );

}


// ============================================================
// OVERLAY
// ============================================================

function setupOverlay() {

    overlay.width =
        video.videoWidth;

    overlay.height =
        video.videoHeight;

}


// ============================================================
// PREPROCESS IMAGE
// ============================================================

function preprocess() {

    processingCtx.clearRect(
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE
    );

    const videoWidth =
        video.videoWidth;

    const videoHeight =
        video.videoHeight;


    const scale =
        Math.min(
            INPUT_SIZE / videoWidth,
            INPUT_SIZE / videoHeight
        );


    const newWidth =
        videoWidth * scale;

    const newHeight =
        videoHeight * scale;


    const offsetX =
        (INPUT_SIZE - newWidth) / 2;

    const offsetY =
        (INPUT_SIZE - newHeight) / 2;


    processingCtx.fillStyle =
        "#808080";

    processingCtx.fillRect(
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE
    );


    processingCtx.drawImage(
        video,

        0,
        0,
        videoWidth,
        videoHeight,

        offsetX,
        offsetY,
        newWidth,
        newHeight
    );


    const imageData =
        processingCtx.getImageData(
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
        );


    const data =
        imageData.data;


    for (
        let i = 0;
        i < CHANNEL_SIZE;
        i++
    ) {
        const px = i * 4;
        PRE_ALLOCATED_FLOAT_DATA[i] = data[px] / 255.0;
        PRE_ALLOCATED_FLOAT_DATA[CHANNEL_SIZE + i] = data[px + 1] / 255.0;
        PRE_ALLOCATED_FLOAT_DATA[CHANNEL_SIZE * 2 + i] = data[px + 2] / 255.0;
    }


    return {
        tensor: new ort.Tensor(
            "float32",
            PRE_ALLOCATED_FLOAT_DATA,
            [
                1,
                3,
                INPUT_SIZE,
                INPUT_SIZE
            ]
        ),

        scale: scale,

        offsetX: offsetX,

        offsetY: offsetY

    };

}


// ============================================================
// INFERENCE
// ============================================================

async function runInference() {

    const prep =
        preprocess();


    const inputName =
        session.inputNames[0];


    const feeds = {};

    feeds[inputName] =
        prep.tensor;


    const results =
        await session.run(feeds);


    const outputName =
        session.outputNames[0];


    const output =
        results[outputName];


    return {
        output: output,
        scale: prep.scale,
        offsetX: prep.offsetX,
        offsetY: prep.offsetY
    };

}


// ============================================================
// YOLO POSTPROCESS
// ============================================================

function postprocess(
    output,
    scale,
    offsetX,
    offsetY
) {

    const data =
        output.data;


    const dims =
        output.dims;

    const numChannels =
        dims[1];

    const numBoxes =
        dims[2];

    const numClasses =
        numChannels - 4;


    const candidates = [];


    for (
        let i = 0;
        i < numBoxes;
        i++
    ) {

        const cx =
            data[
                0 * numBoxes + i
            ];

        const cy =
            data[
                1 * numBoxes + i
            ];

        const w =
            data[
                2 * numBoxes + i
            ];

        const h =
            data[
                3 * numBoxes + i
            ];


        let bestClass = -1;

        let bestScore = 0;


        for (
            let c = 0;
            c < numClasses;
            c++
        ) {

            const score =
                data[
                    (4 + c) *
                    numBoxes + i
                ];


            if (
                score > bestScore
            ) {

                bestScore =
                    score;

                bestClass =
                    c;

            }

        }


        if (
            bestScore <
            CONF_THRESHOLD
        ) {

            continue;

        }


        let x1 =
            cx - w / 2;

        let y1 =
            cy - h / 2;

        let x2 =
            cx + w / 2;

        let y2 =
            cy + h / 2;


        x1 =
            (x1 - offsetX)
            / scale;

        y1 =
            (y1 - offsetY)
            / scale;

        x2 =
            (x2 - offsetX)
            / scale;

        y2 =
            (y2 - offsetY)
            / scale;


        x1 =
            Math.max(
                0,
                Math.min(
                    video.videoWidth,
                    x1
                )
            );

        y1 =
            Math.max(
                0,
                Math.min(
                    video.videoHeight,
                    y1
                )
            );

        x2 =
            Math.max(
                0,
                Math.min(
                    video.videoWidth,
                    x2
                )
            );

        y2 =
            Math.max(
                0,
                Math.min(
                    video.videoHeight,
                    y2
                )
            );


        candidates.push({

            x1: x1,

            y1: y1,

            x2: x2,

            y2: y2,

            score: bestScore,

            classId: bestClass

        });

    }


    const rawDetections = nms(
        candidates,
        IOU_THRESHOLD
    );

    // ============================================================
    // THỰC THI THUẬT TOÁN LÀM MƯỢT (ANTI-JITTER SMOOTHING)
    // ============================================================
    return smoothDetections(rawDetections);
}


// ============================================================
// SMOOTHING & TRACKING HELPER
// ============================================================

function calculateIoU(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);

    const intersectionWidth = Math.max(0, x2 - x1);
    const intersectionHeight = Math.max(0, y2 - y1);
    const intersection = intersectionWidth * intersectionHeight;

    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    const union = areaA + areaB - intersection;

    if (union <= 0) return 0;
    return intersection / union;
}

function smoothDetections(currentDetections) {
    const smoothed = [];

    currentDetections.forEach(current => {
        // Tìm xem vật thể hiện tại có khớp với vật thể nào ở khung hình trước không (cùng class và IoU > 0.3)
        let matchedPrevious = null;
        let bestIoU = 0.3;

        previousDetections.forEach(prev => {
            if (prev.classId === current.classId) {
                const iou = calculateIoU(current, prev);
                if (iou > bestIoU) {
                    bestIoU = iou;
                    matchedPrevious = prev;
                }
            }
        });

        if (matchedPrevious) {
            // Áp dụng thuật toán EMA (Exponential Moving Average) để làm mượt tọa độ
            const x1 = SMOOTHING_ALPHA * current.x1 + (1 - SMOOTHING_ALPHA) * matchedPrevious.x1;
            const y1 = SMOOTHING_ALPHA * current.y1 + (1 - SMOOTHING_ALPHA) * matchedPrevious.y1;
            const x2 = SMOOTHING_ALPHA * current.x2 + (1 - SMOOTHING_ALPHA) * matchedPrevious.x2;
            const y2 = SMOOTHING_ALPHA * current.y2 + (1 - SMOOTHING_ALPHA) * matchedPrevious.y2;
            const score = SMOOTHING_ALPHA * current.score + (1 - SMOOTHING_ALPHA) * matchedPrevious.score;

            smoothed.push({
                x1, y1, x2, y2, score,
                classId: current.classId,
                missingFrames: 0
            });
        } else {
            // Vật thể mới xuất hiện
            smoothed.push({
                ...current,
                missingFrames: 0
            });
        }
    });

    // Giữ lại các vật thể cũ trong một vài khung hình nếu model bị bỏ lỡ chớp nhoáng (tránh nhấp nháy)
    previousDetections.forEach(prev => {
        const found = smoothed.some(curr => curr.classId === prev.classId && calculateIoU(curr, prev) > 0.2);
        if (!found && prev.missingFrames < MAX_MISSING_FRAMES) {
            smoothed.push({
                ...prev,
                missingFrames: prev.missingFrames + 1
            });
        }
    });

    previousDetections = smoothed;
    return smoothed;
}


// ============================================================
// NMS
// ============================================================

function nms(
    boxes,
    threshold
) {

    boxes.sort(
        (a, b) =>
            b.score -
            a.score
    );


    const selected = [];


    while (
        boxes.length > 0
    ) {

        const current =
            boxes.shift();


        selected.push(
            current
        );


        boxes =
            boxes.filter(
                box => {

                    if (
                        box.classId !==
                        current.classId
                    ) {

                        return true;

                    }


                    return (
                        calculateIoU(
                            current,
                            box
                        ) < threshold
                    );

                }
            );

    }


    return selected;

}


// ============================================================
// DRAW DETECTIONS
// ============================================================

function drawDetections(
    detections
) {

    overlayCtx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );


    const scaleX =
        overlay.width /
        video.videoWidth;


    const scaleY =
        overlay.height /
        video.videoHeight;


    overlayCtx.lineWidth = 3;

    overlayCtx.font =
        "bold 16px Arial";


    detections.forEach(
        detection => {

            const x =
                detection.x1 *
                scaleX;

            const y =
                detection.y1 *
                scaleY;

            const width =
                (
                    detection.x2 -
                    detection.x1
                ) *
                scaleX;

            const height =
                (
                    detection.y2 -
                    detection.y1
                ) *
                scaleY;


            const className =
                CLASS_NAMES[
                    detection.classId
                ] ||
                `Class ${detection.classId}`;


            const classColor =
                CLASS_COLORS[
                    detection.classId
                ] || "#00ff00";


            const confidence =
                (
                    detection.score *
                    100
                ).toFixed(1);


            const label =
                `${className} ${confidence}%`;


            overlayCtx.strokeStyle =
                classColor;

            overlayCtx.strokeRect(
                x,
                y,
                width,
                height
            );


            const textWidth =
                overlayCtx.measureText(
                    label
                ).width;


            overlayCtx.fillStyle =
                classColor;

            overlayCtx.fillRect(
                x,
                Math.max(
                    0,
                    y - 24
                ),
                textWidth + 12,
                24
            );


            overlayCtx.fillStyle =
                "#000";

            overlayCtx.fillText(
                label,
                x + 6,
                Math.max(
                    17,
                    y - 7
                )
            );

        }
    );


    objectsElement.textContent =
        detections.length;


    if (
        detections.length === 0
    ) {

        detectionsElement.textContent =
            "No detection";

        return;

    }


    detectionsElement.innerHTML =
        detections
            .map(
                detection => {

                    const className =
                        CLASS_NAMES[
                            detection.classId
                        ] ||
                        `Class ${detection.classId}`;


                    const confidence =
                        (
                            detection.score *
                            100
                        ).toFixed(1);


                    return `
                        <div>
                            🌸
                            <strong>
                                ${className}
                            </strong>
                            -
                            ${confidence}%
                        </div>
                    `;

                }
            )
            .join("");

}


// ============================================================
// FPS
// ============================================================

function updateFPS() {

    frameCounter++;

    const now =
        performance.now();


    const elapsed =
        now - lastTime;


    if (
        elapsed >= 1000
    ) {

        fps =
            frameCounter *
            1000 /
            elapsed;


        fpsElement.textContent =
            fps.toFixed(1);


        frameCounter = 0;

        lastTime = now;

    }

}


// ============================================================
// MAIN REALTIME LOOP
// ============================================================

let lastInferenceTime = 0;

const INFERENCE_INTERVAL = 100;

async function inferenceLoop(timestamp) {

    if (!running) {
        return;
    }

    if (video.readyState >= 2) {

        if (
            timestamp - lastInferenceTime >=
            INFERENCE_INTERVAL
        ) {

            lastInferenceTime = timestamp;

            try {

                const result =
                    await runInference();

                const detections =
                    postprocess(
                        result.output,
                        result.scale,
                        result.offsetX,
                        result.offsetY
                    );

                drawDetections(
                    detections
                );

                updateFPS();

            } catch (error) {

                console.error(
                    "Inference error:",
                    error
                );

                setStatus(
                    "Inference error - see Console"
                );
            }
        }
    }

    requestAnimationFrame(
        inferenceLoop
    );
}


// ============================================================
// BUTTONS
// ============================================================

loadButton.addEventListener(
    "click",
    loadModel
);


cameraButton.addEventListener(
    "click",
    startCamera
);


stopButton.addEventListener(
    "click",
    stopCamera
);


// ============================================================
// INITIAL STATUS
// ============================================================

setStatus(
    "Ready - load model"
);