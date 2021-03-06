/* https://github.com/danijel3/audio_gui/blob/master/static/main.js */
/* https://github.com/cwilso/AudioRecorder/blob/master/js/main.js */
/* updated on 2020.12.05

/* Copyright 2013 Chris Wilson
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

window.AudioContext = window.AudioContext || window.webkitAudioContext;

var audioContext = new AudioContext();
var audioInput = null,
    realAudioInput = null,
    inputPoint = null,
    audioRecorder = null;
var rafID = null;
var analyserContext = null;
var canvasWidth, canvasHeight;
var recIndex = 0;


function gotBuffers(buffers) {
    audioRecorder.exportMonoWAV(doneEncoding);
}

function convert_percentage(input){
    return parseFloat(input*100).toFixed(2)+"%";
}

function isNumber(value) {
    if ((undefined === value) || (null === value)) {
        return false;
    }
    if (typeof value == 'number') {
        return true;
    }
    return !isNaN(value - 0);
}

function isObject(obj)
{
    return obj !== undefined && obj !== null && obj.constructor == Object;
}


/* 20201212 1604 - changed /audio to /recorder */
function doneEncoding(soundBlob) {
    // fetch('/audio', {method: "POST", body: soundBlob}).then(response => $('#output').text(response.text()))
    
    var spinner = document.getElementById("spinner");
    spinner.style.display = "block";
    fetch('/recorder', {method: "POST", body: soundBlob}).then(response => response.text().then(text => {
        spinner.style.display = "none";
        res = JSON.parse(text);
        document.getElementById('output').value = res[0];
        console.log(res[1]);
        data = res[1]

        t_data = [];

        for (label in data) {
            output = data[label]
            if (isObject(output)) {
                new_output = ""
                for (emotion in output) {
                    if (isNumber(output[emotion])){
                        prob = convert_percentage(output[emotion]);
                        new_output += emotion + " is " + prob.toString()+"\n";
                    } else {
                        new_output += emotion + " is " + output[emotion]+"\n";
                    }
                }
                var new_row = {"label": label, "output": new_output};
            } else {
                var new_row = {"label": label, "output": JSON.stringify(output)};
            };
            
            t_data.push(new_row);  
          }

        console.log(t_data);


        //initialize table
        var table = new Tabulator("#prediction-table", {
            data:t_data, //assign data to table
            layout:"fitDataStretch",
            // autoColumns:true, //create columns from data field names
            columns:[
                {title:"Label", field:"label"}, // formatter:"uppercase", formatter:"bold"
                {title:"Output", field:"output", formatter:"textarea"}
                ],
        });
        // document.getElementById("prediction").innerHTML= JSON.stringify(res[1]);

    }));
    recIndex++;
}

function stopRecording() {
    // stop recording
    audioRecorder.stop();
    document.getElementById('stop').disabled = true;
    document.getElementById('start').removeAttribute('disabled');
    audioRecorder.getBuffers(gotBuffers);
}

function startRecording() {

    // start recording
    if (!audioRecorder)
        return;
    document.getElementById('start').disabled = true;
    document.getElementById('stop').removeAttribute('disabled');
    audioRecorder.clear();
    audioRecorder.record();
}

function convertToMono(input) {
    var splitter = audioContext.createChannelSplitter(2);
    var merger = audioContext.createChannelMerger(2);

    input.connect(splitter);
    splitter.connect(merger, 0, 0);
    splitter.connect(merger, 0, 1);
    return merger;
}

function cancelAnalyserUpdates() {
    window.cancelAnimationFrame(rafID);
    rafID = null;
}

function updateAnalysers(time) {
    if (!analyserContext) {
        var canvas = document.getElementById("analyser");
        canvasWidth = canvas.width;
        canvasHeight = canvas.height;
        analyserContext = canvas.getContext('2d');
    }

    // analyzer draw code here
    {
        var SPACING = 3;
        var BAR_WIDTH = 1;
        var numBars = Math.round(canvasWidth / SPACING);
        var freqByteData = new Uint8Array(analyserNode.frequencyBinCount);

        analyserNode.getByteFrequencyData(freqByteData);

        analyserContext.clearRect(0, 0, canvasWidth, canvasHeight);
        analyserContext.fillStyle = '#F6D565';
        analyserContext.lineCap = 'round';
        var multiplier = analyserNode.frequencyBinCount / numBars;

        // Draw rectangle for each frequency bin.
        for (var i = 0; i < numBars; ++i) {
            var magnitude = 0;
            var offset = Math.floor(i * multiplier);
            // gotta sum/average the block, or we miss narrow-bandwidth spikes
            for (var j = 0; j < multiplier; j++)
                magnitude += freqByteData[offset + j];
            magnitude = magnitude / multiplier;
            var magnitude2 = freqByteData[i * multiplier];
            analyserContext.fillStyle = "hsl( " + Math.round((i * 360) / numBars) + ", 100%, 50%)";
            analyserContext.fillRect(i * SPACING, canvasHeight, BAR_WIDTH, -magnitude);
        }
    }

    rafID = window.requestAnimationFrame(updateAnalysers);
}

function toggleMono() {
    if (audioInput != realAudioInput) {
        audioInput.disconnect();
        realAudioInput.disconnect();
        audioInput = realAudioInput;
    } else {
        realAudioInput.disconnect();
        audioInput = convertToMono(realAudioInput);
    }

    audioInput.connect(inputPoint);
}

function gotStream(stream) {
    document.getElementById('start').removeAttribute('disabled');

    inputPoint = audioContext.createGain();

    // Create an AudioNode from the stream.
    realAudioInput = audioContext.createMediaStreamSource(stream);
    audioInput = realAudioInput;
    audioInput.connect(inputPoint);

//    audioInput = convertToMono( input );

    analyserNode = audioContext.createAnalyser();
    analyserNode.fftSize = 2048;
    inputPoint.connect(analyserNode);

    audioRecorder = new Recorder(inputPoint);

    zeroGain = audioContext.createGain();
    zeroGain.gain.value = 0.0;
    inputPoint.connect(zeroGain);
    zeroGain.connect(audioContext.destination);
    updateAnalysers();
}

function initAudio() {
    if (!navigator.getUserMedia)
        navigator.getUserMedia = navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
    if (!navigator.cancelAnimationFrame)
        navigator.cancelAnimationFrame = navigator.webkitCancelAnimationFrame || navigator.mozCancelAnimationFrame;
    if (!navigator.requestAnimationFrame)
        navigator.requestAnimationFrame = navigator.webkitRequestAnimationFrame || navigator.mozRequestAnimationFrame;

    navigator.getUserMedia(
        {
            "audio": {
                "mandatory": {
                    "googEchoCancellation": "false",
                    "googAutoGainControl": "false",
                    "googNoiseSuppression": "false",
                    "googHighpassFilter": "false"
                },
                "optional": []
            },
        }, gotStream, function (e) {
            alert('Error getting audio');
            console.log(e);
        });
}

window.addEventListener('load', initAudio);

function unpause() {
    document.getElementById('init').style.display = 'none';
    audioContext.resume().then(() => {
        console.log('Playback resumed successfully');
    });
}