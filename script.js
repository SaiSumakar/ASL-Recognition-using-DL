// Scroll to the second section when the "Get Started" button is clicked
document.getElementById("getStarted").addEventListener("click", function () {
    document.getElementById("secondSection").scrollIntoView({ behavior: "smooth" });
  });
  
  
  // Get DOM elements
  const videoElement = document.getElementById('video');
  const startButton = document.getElementById('start');
  const stopButton = document.getElementById('stop');
  
  // To store the MediaStream object
  let mediaStream = null;
  
  // Start the camera
  startButton.addEventListener('click', async () => {
      try {
          // Access the user's camera
          mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
  
          // Set the video element source to the camera feed
          videoElement.srcObject = mediaStream;
  
          // Enable the Stop button
          stopButton.disabled = false;
          startButton.disabled = true;
      } catch (error) {
          console.error('Error accessing camera:', error);
          alert('Unable to access the camera. Please check permissions or device settings.');
      }
  });
  
  // Stop the camera
  stopButton.addEventListener('click', () => {
      if (mediaStream) {
          // Stop all tracks (video feed)
          mediaStream.getTracks().forEach(track => track.stop());
          mediaStream = null;
  
          // Disable the Stop button
          stopButton.disabled = true;
          startButton.disabled = false;
      }
  });
  
  
  // Function to update the text dynamically
  function updateDetectedText(detectedText) {
    const textContainer = document.getElementById("dynamic-text");
    textContainer.textContent = detectedText; // Update the content dynamically
  }
  
  
  
  const video = document.getElementById('video');
  
  function captureFrame() {
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
  
      // Draw the current video frame to the canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
  
      // Convert the canvas to a base64-encoded string
      return canvas.toDataURL('image/jpeg').split(',')[1]; // Remove the header
  }
  
  async function sendFrame() {
      const frame = captureFrame();
  
      try {
          const response = await fetch('http://127.0.0.1:5000/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ frame: frame }),
          });
  
          const data = await response.json();
          if (data.error) {
              console.error('Error from backend:', data.error);
          } else {
              updateDetectedText(data.prediction); // Update detected text
          }
      } catch (error) {
          console.error('Error sending frame:', error);
      }
  }
  
  // Call this function at regular intervals while video is running
  setInterval(sendFrame, 500);  // Adjust interval as needed
  
  
  