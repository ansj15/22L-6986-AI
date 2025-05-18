document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');
    const uploadText = document.getElementById('uploadText');
    const detectButton = document.getElementById('detectButton');
    const result = document.getElementById('result');
    const prediction = document.getElementById('prediction');
    const confidenceLevel = document.getElementById('confidenceLevel');

    // Handle drag and drop events
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#4CAF50';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = '#ccc';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#ccc';
        const file = e.dataTransfer.files[0];
        handleFile(file);
    });

    // Handle click to upload
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        handleFile(file);
    });

    // Handle the selected file
    function handleFile(file) {
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
                uploadText.style.display = 'none';
                detectButton.disabled = false;
            };
            reader.readAsDataURL(file);
        } else {
            alert('Please upload an image file.');
        }
    }

    // Handle detect button click
    detectButton.addEventListener('click', async () => {
        const file = fileInput.files[0] || new File([dataURLtoBlob(preview.src)], 'image.jpg');
        const formData = new FormData();
        formData.append('file', file);

        try {
            detectButton.disabled = true;
            detectButton.textContent = 'Detecting...';

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            result.style.display = 'block';
            prediction.textContent = `Detected Instrument: ${data.prediction}`;
            confidenceLevel.style.width = `${data.confidence * 100}%`;
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            detectButton.disabled = false;
            detectButton.textContent = 'Detect Instrument';
        }
    });

    // Helper function to convert Data URL to Blob
    function dataURLtoBlob(dataURL) {
        const arr = dataURL.split(',');
        const mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n);
        while (n--) {
            u8arr[n] = bstr.charCodeAt(n);
        }
        return new Blob([u8arr], { type: mime });
    }
}); 