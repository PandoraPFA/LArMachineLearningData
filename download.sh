function download() {
   wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$1' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
}

# Google Drive links have the form https://drive.google.com/file/d/FILEID/view?usp=sharing
# You want the FILEID
# Note: For some reason the above function doesn't work for the files, so you need to set the two &id=FILEID instances in the link

### PandoraMVAData
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraMVAData
cd $MY_TEST_AREA/LArMachineLearningData/PandoraMVAData

# MicroBooNE
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1b3m9Glj1Qjx5tnSdvqLIBNFBWpH8NrvX' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1b3m9Glj1Qjx5tnSdvqLIBNFBWpH8NrvX" -O PandoraSvm_v03_11_00.xml && rm -rf /tmp/cookies.txt

# DUNE

wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1i5mi545-NQU15raIyYOwbWY8VQuCv8Ox' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1i5mi545-NQU15raIyYOwbWY8VQuCv8Ox" -O PandoraBdt_BeamParticleId_ProtoDUNESP_v03_26_00.xml && rm -rf /tmp/cookies.txt
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1eeiScUaLjEoACxCyb73JwGnuMsJ1Y4Iq' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1eeiScUaLjEoACxCyb73JwGnuMsJ1Y4Iq" -O PandoraBdt_PfoCharacterisation_ProtoDUNESP_v03_26_00.xml && rm -rf /tmp/cookies.txt
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=12cl3gpijkcpIIZZGeThtH2e4Vzvf6Xqu' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12cl3gpijkcpIIZZGeThtH2e4Vzvf6Xqu" -O PandoraBdt_Vertexing_ProtoDUNESP_v03_26_00.xml && rm -rf /tmp/cookies.txt

wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1oMDHwtOapNcs8m4H29MpQkiKbFo0rWbT' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1oMDHwtOapNcs8m4H29MpQkiKbFo0rWbT" -O PandoraBdt_PfoCharacterisation_DUNEFD_v03_26_00.xml && rm -rf /tmp/cookies.txt
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RasATFZxy-vwai6mJCmvAew8QQueKbj2' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1RasATFZxy-vwai6mJCmvAew8QQueKbj2" -O PandoraBdt_Vertexing_DUNEFD_v03_26_00.xml && rm -rf /tmp/cookies.txt

# SBND
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1A4za5GX8Y23kcUphMOofTDuOpAdvTOU6' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1A4za5GX8Y23kcUphMOofTDuOpAdvTOU6" -O PandoraBdt_v08_33_00_SBND.xml && rm -rf /tmp/cookies.txt

### PandoraNetworkData
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraNetworkData
cd $MY_TEST_AREA/LArMachineLearningData/PandoraNetworkData

# DUNE
download "1EOmyofKFl9NAZqRh-z34bY7-M9lXvG9U" "PandoraUnet_TSID_DUNEFD_U_v03_25_00.pt"
download "1NqeKHnCkWdTIKw6CLqUZp4l-q7EIkAQa" "PandoraUnet_TSID_DUNEFD_V_v03_25_00.pt"
download "1kWfT8d4GtlMsoxh6cH0quIUNRdHSClhj" "PandoraUnet_TSID_DUNEFD_W_v03_25_00.pt"

