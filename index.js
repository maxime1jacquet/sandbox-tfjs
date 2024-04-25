const tf = require("@tensorflow/tfjs-node");
const express = require("express");
const app = express();

var model;

async function init() {
  let array = [
    { score: 1, letter: "A" },
    { score: 2, letter: "B" },
    { score: 3, letter: "C" },
    { score: 4, letter: "D" },
    { score: 5, letter: "E" },
    { score: 6, letter: "F" },
    { score: 7, letter: "G" },
    { score: 8, letter: "H" },
    { score: 9, letter: "I" },
    { score: 10, letter: "J" },
    { score: 11, letter: "K" },
    { score: 12, letter: "L" },
    { score: 13, letter: "M" },
    { score: 14, letter: "N" },
    { score: 15, letter: "O" },
    { score: 16, letter: "P" },
    { score: 17, letter: "Q" },
    { score: 18, letter: "R" },
    { score: 19, letter: "S" },
    { score: 20, letter: "T" },
    { score: 21, letter: "U" },
    { score: 22, letter: "V" },
    { score: 23, letter: "W" },
    { score: 24, letter: "X" },
    { score: 25, letter: "Y" },
    { score: 26, letter: "Z" },
  ];
  let data = array.map((item) => {
    let encoded = Array(26).fill(0);
    encoded[item.letter.charCodeAt(0) - "A".charCodeAt(0)] = 1;
    return encoded;
  });
  let scores = tf.tensor(array.map((item) => item.score));
  let tensorData = tf.tensor(data);

  model = tf.sequential();
  model.add(
    tf.layers.dense({ units: 100, inputShape: [26], activation: "relu" })
  );
  model.add(tf.layers.dense({ units: 50, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1 }));
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });
  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });
  await model.fit(tensorData, scores, { epochs: 150 });
}

async function predict(letterparams) {
  let encoded = Array(26).fill(0);
  encoded[letterparams.charCodeAt(0) - "A".charCodeAt(0)] = 1;
  let tensor = tf.tensor([encoded]);
  let score = model.predict(tensor);
  let dataValue = await score.data();

  console.log(`---------------------------------`);
  console.log(`---------------------------------`);
  console.log(`---------------------------------`);
  console.log(`Le score de la lettre ${letterparams} est ${dataValue}`);

  return dataValue;
}

// Définir une route pour la page d'accueil
app.get("/", async (req, res) => {
  const letter = req.query.letter ?? "A";
  const predictScore = await predict(letter);

  res.send(`Predict letter ${letter} : ${predictScore}`);
});

// Démarrer le serveur sur le port 3000
app.listen(3000, async () => {
  await init();
  console.log("Application démarrée sur le port 3000");
});
