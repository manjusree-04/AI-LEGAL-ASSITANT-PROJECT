// Voice Assistant JavaScript
let mediaRecorder;
let audioChunks = [];

// Start recording
function startVoiceRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            
            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });
            
            document.getElementById('voiceBtn').textContent = '🔴 Stop';
        });
}

// Stop recording and send to backend
function stopVoiceRecording() {
    mediaRecorder.stop();
    mediaRecorder.addEventListener("stop", () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        audioChunks = [];
        
        // Send to backend
        const formData = new FormData();
        formData.append('audio', audioBlob, 'query.wav');
        
        fetch('/voice_to_text', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.text) {
                document.getElementById('queryInput').value = data.text;
                // Auto-submit query
                sendQuery(data.text);
            }
        });
    });
    
    document.getElementById('voiceBtn').textContent = '🎤 Voice';
}

// Text to speech
function speakResponse(text) {
    fetch('/text_to_speech', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text})
    })
    .then(response => response.blob())
    .then(blob => {
        const audio = new Audio(URL.createObjectURL(blob));
        audio.play();
    });
}

// Toggle voice recording
function toggleVoiceRecording() {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') {
        startVoiceRecording();
    } else {
        stopVoiceRecording();
    }
}