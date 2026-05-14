const express = require('express');
const bodyParser = require('body-parser');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const port = 3000;

app.use(bodyParser.json());
app.use(express.static('src')); // Change the folder name to 'src'

// GET route to serve the HTML page
app.get('/', (req, res) => {
    // Send the HTML file when someone accesses the root URL
    res.sendFile(path.join(__dirname, 'src', 'index.html')); // Adjust the path based on your file structure
});

app.post('/predict', (req, res) => {
    console.log("received predict request");
    const inputData = req.body;
    console.log("Input Data:", inputData);

    let responseSent = false; // Flag to track if a response has been sent

    const venvPython = path.join(__dirname, '.venv', 'bin', 'python3');
    const pythonProcess = spawn(venvPython, ['backend.py', JSON.stringify(inputData)]);

    pythonProcess.stdout.on('data', (data) => {
        console.log("Received data from Python process:", data.toString().trim());
        const predictions = JSON.parse(data.toString().trim());
        console.log("Predictions:", predictions);
        if (!responseSent) {
            res.send({ predictions });
            responseSent = true;
        }
    });

    pythonProcess.stderr.on('data', (data) => {
        const errorMessage = data.toString().trim();
        console.error("Python Process Error:", errorMessage);
        if (!responseSent) {
            res.status(500).send({ error: "Internal Server Error" });
            responseSent = true;
        }
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python process closed with code ${code}`);
    });
});


app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
