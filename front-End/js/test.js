function submition() {
  age = document.getElementById("age").value;
  sex = document.getElementById("sex").value;
  cp = document.getElementById("cp").value;
  trestbps = document.getElementById("trestbps").value;
  chol = document.getElementById("chol").value;
  fbs = document.getElementById("fbs").value;
  restecg = document.getElementById("restecg").value;
  thalach = document.getElementById("thalach").value;
  exang = document.getElementById("exang").value;
  oldpeak = document.getElementById("oldpeak").value;
  slope = document.getElementById("slope").value;
  ca = document.getElementById("ca").value;
  thal = document.getElementById("thal").value;
  axios
    .get(
      "http://127.0.0.1:5000/getresults?age=" +
        age +
        "&sex=" +
        sex +
        "&cp=" +
        cp +
        "&trestbps=" +
        trestbps +
        "&chol=" +
        chol +
        "&fbs=" +
        fbs +
        "&restecg=" +
        restecg +
        "&thalach=" +
        thalach +
        "&exang=" +
        exang +
        "&oldpeak=" +
        oldpeak +
        "&slope=" +
        slope +
        "&ca=0" +
        ca +
        "&thal=" +
        thal,
      function (req, res) {
        res.header("Access-Control-Allow-Origin", "*");
      }
    )
    .then((result) => {
      document.getElementById("result").innerHTML = JSON.stringify(
        result.data.Prediction
      );
    })
    .catch((err) => console.error(err));
}
