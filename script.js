import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";

// grab a reference to the mnist input values (pixel data)
const INPUTS = TRAINING_DATA.inputs;

// grab reference to the mnist output values.
const OUTPUTS = TRAINING_DATA.outputs;

// shuffle the two arrays in the same way so inputs still match outputs indexes.
tf.util.shuffleCombo(INPUTS, OUTPUTS);

// input feature array is 1 dimensional
const INPUTS_TENSOR = tf.tensor2d(INPUTS);

// output feature array is 1 dimensional
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

// create and define model architecture
const model = tf.sequential();

model.add(
  tf.layers.dense({ inputShape: [784], units: 32, activation: "relu" })
);
model.add(tf.layers.dense({ units: 16, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

model.summary();

train();

async function train() {
  // compile the model with the defined optimizer and specify our loss function to use.
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
    shuffle: true, // ensure data is shuffled again before using ech epoch
    validationSplit: 0.2,
    batchSize: 512, // update weights after every 512 samples
    epochs: 50, // go over the data 50 times
    callbacks: { onEpochEnd: logProgress },
  });

  OUTPUTS_TENSOR.dispose();
  INPUTS_TENSOR.dispose();
  evaluate(); // once trained, the model is evaluated.
}

function logProgress(epoch, logs) {
  console.log("Data for epoch " + epoch, logs);
}

const PREDICTION_ELEMENT = document.getElementById("prediction");

function evaluate() {
  const OFFSET = Math.floor(Math.random() * INPUTS.length); // select random from all example inputs

  let answer = tf.tidy(function () {
    let newInput = tf.tensor1d(INPUTS[OFFSET]);

    let output = model.predict(newInput.expandDims());
    output.print();
    return output.squeeze().argMax();
  });

  answer.array().then(function (index) {
    PREDICTION_ELEMENT.innerText = index;
    PREDICTION_ELEMENT.setAttribute(
      "class",
      index === OUTPUTS[OFFSET] ? "correct" : "wrong"
    );
    answer.dispose();
    drawImage(INPUTS[OFFSET]);
  });
}

const CANVAS = document.getElementById("canvas");

const CTX = CANVAS.getContext("2d");

const interval = 2000;

function drawImage(digit) {
  var imageData = CTX.getImageData(0, 0, 28, 28);

  for (let i = 0; i < digit.length; i++) {
    imageData.data[i * 4] = digit[i] * 255; // red channel
    imageData.data[i * 4 + 1] = digit[i] * 255; // green channel
    imageData.data[i * 4 + 2] = digit[i] * 255; // blue channel
    imageData.data[i * 4 + 3] = 255; // alpha channel
  }

  // render the updated array of data to the canvas itself.
  CTX.putImageData(imageData, 0, 0);

  // perform a new classification after a certain interval
  setTimeout(evaluate, interval);
}
