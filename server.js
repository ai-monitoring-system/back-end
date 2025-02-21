// server.js
const express = require('express');
const { spawn } = require('child_process');
const cors = require('cors'); // <--- import cors

const app = express();

app.use(cors());             // <--- enable CORS
app.use(express.json());

// POST /start-transceiver
app.post('/start-transceiver', (req, res) => {
  const { userId } = req.body;
  console.log('Starting transceiver with userId:', userId);

  const pythonProcess = spawn('python', ['transceiver.py', userId]);

  pythonProcess.stdout.on('data', (data) => {
    console.log(`transceiver stdout: ${data}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`transceiver stderr: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`transceiver.py exited with code ${code}`);
  });

  res.status(200).send('Transceiver started');
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});
