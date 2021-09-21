function download() {
   wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$1' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2 && rm -rf /tmp/cookies.txt
}

# Google Drive links have the form https://drive.google.com/file/d/FILEID/view?usp=sharing
# You want the FILEID
# Note: For some reason the above function doesn't work for the files, so you need to set the two &id=FILEID instances in the link

# PandoraMVAData
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraMVAData
cd $MY_TEST_AREA/LArMachineLearningData/PandoraMVAData
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1b3m9Glj1Qjx5tnSdvqLIBNFBWpH8NrvX' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1b3m9Glj1Qjx5tnSdvqLIBNFBWpH8NrvX" -O PandoraSvm_v03_11_00.xml && rm -rf /tmp/cookies.txt
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-MIy2zrx1HWJDvtxFwljd392sIotSrFX' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-MIy2zrx1HWJDvtxFwljd392sIotSrFX" -O PandoraBdt_v03_15_02.xml && rm -rf /tmp/cookies.txt
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BcZk3aBWj_VEiGTMZchtwM73YT5yWEaW' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BcZk3aBWj_VEiGTMZchtwM73YT5yWEaW" -O PandoraBdt_v03_20_00.xml && rm -rf /tmp/cookies.txt
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xUnLN-oWg-c5u4docaxTijMVFIq3faga' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xUnLN-oWg-c5u4docaxTijMVFIq3faga" -O PandoraBdt_ProtoDUNE_v03_20_00.xml && rm -rf /tmp/cookies.txt
wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1A4za5GX8Y23kcUphMOofTDuOpAdvTOU6' -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1A4za5GX8Y23kcUphMOofTDuOpAdvTOU6" -O PandoraBdt_v08_33_00_SBND.xml && rm -rf /tmp/cookies.txt

# PandoraNetworkData
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraNetworkData
cd $MY_TEST_AREA/LArMachineLearningData/PandoraNetworkData
download "1EOmyofKFl9NAZqRh-z34bY7-M9lXvG9U" "PandoraUnet_TSID_DUNEFD_U_v03_25_00.pt"
download "1NqeKHnCkWdTIKw6CLqUZp4l-q7EIkAQa" "PandoraUnet_TSID_DUNEFD_V_v03_25_00.pt"
download "1kWfT8d4GtlMsoxh6cH0quIUNRdHSClhj" "PandoraUnet_TSID_DUNEFD_W_v03_25_00.pt"

