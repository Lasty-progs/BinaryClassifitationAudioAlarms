You may read about this research on this <a href="https://paperswithcode.com/">paper</a>

<pre><h3>Instruction for setup and start:<h3>
  <t><h4>1.Clone repository and download dataset from zenodo:</h4>
      1.1 comand: "git clone https://github.com/Lasty-progs/BinaryClassifitationAudioAlarms.git"
      1.2 go to the https://zenodo.org/records/4060432 and download: <br>
        - FSD50K.ground_truth.zip <br>
        - FSD50K.eval_audio.zip <br>
        - FSD50K.dev_audio.zip <br>
      Unpack files for FSD50K/ground_truth, FSD50K/eval_audio and FSD50K/dev_audio in Project folder
  <h4>2.Install all dependencies:</h4>
      Make sure that Python==3.10, Cuda==12.3 and cuDNN==8.9.7 was installed.
      Create and activate venv and run: pip install -r req.txt
  <h4>3.Run data_formater.py for create temp file with spectrogramms
  <h4>4.Run neural.py for train and predict values for evaluation data in submission.txt file
    Metrics will overwrite my results in metrics folder<pre>