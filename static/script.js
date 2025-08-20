document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const audioFile = document.getElementById('audio-file');
    const predictionResult = document.getElementById('prediction-result');
    const recordBtn = document.getElementById('record-btn');
    const stopBtn = document.getElementById('stop-btn');
    const audioPlayer = document.getElementById('audio-player');
    const predictRecordingBtn = document.getElementById('predict-recording-btn');

    let mediaRecorder;
    let audioChunks = [];

    // --- New Helper Function to convert Blob to WAV ---
    async function blobToWav(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => {
                const arrayBuffer = reader.result;
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                audioContext.decodeAudioData(arrayBuffer, (audioBuffer) => {
                    const wavBuffer = audioBufferToWav(audioBuffer);
                    resolve(new Blob([wavBuffer], { type: 'audio/wav' }));
                }, reject);
            };
            reader.onerror = reject;
            reader.readAsArrayBuffer(blob);
        });
    }

    function audioBufferToWav(buffer) {
        let numOfChan = buffer.numberOfChannels,
            len = buffer.length * numOfChan * 2 + 44,
            buffer_out = new ArrayBuffer(len),
            view = new DataView(buffer_out),
            channels = [],
            i,
            sample,
            offset = 0,
            pos = 0;

        // write WAV header
        setUint32(0x46464952); // "RIFF"
        setUint32(len - 8); // file length - 8
        setUint32(0x45564157); // "WAVE"

        setUint32(0x20746d66); // "fmt " chunk
        setUint32(16); // length = 16
        setUint16(1); // PCM (uncompressed)
        setUint16(numOfChan);
        setUint32(buffer.sampleRate);
        setUint32(buffer.sampleRate * 2 * numOfChan); // avg. bytes/sec
        setUint16(numOfChan * 2); // block-align
        setUint16(16); // 16-bit
        
        setUint32(0x61746164); // "data" - chunk
        setUint32(len - pos - 4); // chunk length

        // write interleaved data
        for (i = 0; i < buffer.numberOfChannels; i++)
            channels.push(buffer.getChannelData(i));

        while (pos < len) {
            for (i = 0; i < numOfChan; i++) {
                sample = Math.max(-1, Math.min(1, channels[i][offset])); // clamp
                sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0; // scale to 16-bit signed int
                view.setInt16(pos, sample, true); // write 16-bit sample
                pos += 2;
            }
            offset++;
        }
        return buffer_out;

        function setUint16(data) {
            view.setUint16(pos, data, true);
            pos += 2;
        }

        function setUint32(data) {
            view.setUint32(pos, data, true);
            pos += 4;
        }
    }


    // --- Event Listeners ---

    // Handle audio upload
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData();
        formData.append('audio_data', audioFile.files[0]);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        predictionResult.textContent = data.prediction;
    });

    // Handle audio recording
    recordBtn.addEventListener('click', async () => {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();

        recordBtn.disabled = true;
        stopBtn.disabled = false;
        audioChunks = []; // Clear previous recording chunks

        mediaRecorder.addEventListener('dataavailable', event => {
            audioChunks.push(event.data);
        });
    });

    stopBtn.addEventListener('click', () => {
        mediaRecorder.stop();
        recordBtn.disabled = false;
        stopBtn.disabled = true;
        predictRecordingBtn.disabled = false;

        mediaRecorder.addEventListener('stop', () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' }); // Specify mime type
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayer.src = audioUrl;
        });
    });

    // Predict recorded audio
    predictRecordingBtn.addEventListener('click', async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        
        // Convert the blob to WAV before sending
        const wavBlob = await blobToWav(audioBlob);
        
        const formData = new FormData();
        formData.append('audio_data', wavBlob, 'recording.wav');

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        predictionResult.textContent = data.prediction;
    });
});
