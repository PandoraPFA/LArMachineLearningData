function download() {
   curl -L "https://cernbox.cern.ch/s/$1/download" -o $2
}

# CERN box links have the form  curl -L -J -O "https://cernbox.cern.ch/s/FILEID/download"
# You want the FILEID

if [ -z $1 ]
then
  echo "Error: Expected usage source download.sh <experiment> <workflow>"
  echo "   where experiment in [ dune sbnd uboone]"
  return 1
fi

if [ -z $MY_TEST_AREA ]
then
  echo "MY_TEST_AREA is not set, can't download the files"
  return 1
fi

if [ ! -d $MY_TEST_AREA/LArMachineLearningData/ ]
then
  echo "LArMachineLearningData does not exist in MY_TEST_AREA: $MY_TEST_AREA, Not downloading the files"
  return 1
fi

### PandoraMVAData
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraMVAData
cd $MY_TEST_AREA/LArMachineLearningData/PandoraMVAData

# MicroBooNE
if [[ "$1" == "uboone" ]]
then
  download "4gtLmnIUzyhctK0" "PandoraSvm_v03_11_00.xml"
fi

# DUNE

if [[ "$1" == "dune" ]]
then
  download "lh70mgiHZDQgYtd" "PandoraBdt_BeamParticleId_ProtoDUNESP_v03_26_00.xml"
  download "YXWKwk62iRGHEFf" "PandoraBdt_PfoCharacterisation_ProtoDUNESP_v03_26_00.xml"
  download "bLZo7q2VcN1XJCh" "PandoraBdt_Vertexing_ProtoDUNESP_v03_26_00.xml"

  download "HmBOi1FbU3ZY6V2" "PandoraBdt_PfoCharacterisation_DUNEFD_HD_v04_06_00.xml"
  download "Lq6p8HHoUacyVub" "PandoraBdt_PfoCharacterisation_DUNEFD_VD_v04_06_00.xml"
  download "TXa87V98R46HQSw" "PandoraBdt_Vertexing_DUNEFD_v03_27_00.xml"
fi

### PandoraMVAs
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraMVAs
cd $MY_TEST_AREA/LArMachineLearningData/PandoraMVAs

# SBND
if [[ "$1" == "sbnd" ]]
then
  curl_download "fQZBRi1kiWvrJG2" "PandoraBdt_v09_67_00_SBND.xml"
fi

### PandoraNetworkData
mkdir -p $MY_TEST_AREA/LArMachineLearningData/PandoraNetworkData
cd $MY_TEST_AREA/LArMachineLearningData/PandoraNetworkData

# DUNE

if [[ "$1" == "dune" ]]
then
  if [ -z $2 ]
  then
    echo "Error: Expected usage source download.sh dune <workflow>"
    echo "   where workflow in [ lbl atmos lowe ]"
    return 1
  fi

  if [[ "$2" == "lbl" ]]
  then
    download "F3ctEucYgFXhVZF" "PandoraNet_Vertex_DUNEFD_HD_Accel_1_U_v04_06_00.pt"
    download "KJUkvGdM92R2wwZ" "PandoraNet_Vertex_DUNEFD_HD_Accel_1_V_v04_06_00.pt"
    download "HfNMVVyLBCwV8Vp" "PandoraNet_Vertex_DUNEFD_HD_Accel_1_W_v04_06_00.pt"
    download "22o0Go5ayGCHeuV" "PandoraNet_Vertex_DUNEFD_HD_Accel_2_U_v04_06_00.pt"
    download "RqPw75uUa9uln5o" "PandoraNet_Vertex_DUNEFD_HD_Accel_2_V_v04_06_00.pt"
    download "KwB0KWBe38PEGwe" "PandoraNet_Vertex_DUNEFD_HD_Accel_2_W_v04_06_00.pt"

    download "35dY03D7b2e8ACR" "PandoraNet_Vertex_DUNEFD_VD_Accel_1_U_v04_06_00.pt"
    download "ikvEpIBVzM60zlB" "PandoraNet_Vertex_DUNEFD_VD_Accel_1_V_v04_06_00.pt"
    download "yeKvqFqJ3mpUFbz" "PandoraNet_Vertex_DUNEFD_VD_Accel_1_W_v04_06_00.pt"
    download "DOZhZ2qf2D7oQNG" "PandoraNet_Vertex_DUNEFD_VD_Accel_2_U_v04_06_00.pt"
    download "ZZOu0W1aiMMhC2e" "PandoraNet_Vertex_DUNEFD_VD_Accel_2_V_v04_06_00.pt"
    download "tgDnPDVZY1LtJGm" "PandoraNet_Vertex_DUNEFD_VD_Accel_2_W_v04_06_00.pt"

    download "YGX48duFLCI99Li" "PandoraNet_SecVertex_DUNEFD_HD_Accel_1_U_v04_13_00.pt"
    download "pn77VcsHXrTsrkb" "PandoraNet_SecVertex_DUNEFD_HD_Accel_1_V_v04_13_00.pt"
    download "0sMcCSgALuR6RxY" "PandoraNet_SecVertex_DUNEFD_HD_Accel_1_W_v04_13_00.pt"

    download "wqTUQThIpCmtzLo" "PandoraNet_Hierarchy_DUNEFD_HD_S_Class_v014_15_00.pt"
    download "K0PATEqUzsKQodN" "PandoraNet_Hierarchy_DUNEFD_HD_T_Class_v014_15_00.pt"
    download "F3mLtSkMQ3NXw2o" "PandoraNet_Hierarchy_DUNEFD_HD_T_Edge_v014_15_00.pt"
    download "5RYX7iHWPUuVkgK" "PandoraNet_Hierarchy_DUNEFD_HD_TS_Class_v014_15_00.pt"
    download "7SOK7TxRqw6YUYM" "PandoraNet_Hierarchy_DUNEFD_HD_TS_Edge_v014_15_00.pt"
    download "roB3RbWOUW7Qa4N" "PandoraNet_Hierarchy_DUNEFD_HD_TT_Class_v014_15_00.pt"
    download "L0ktaA5faHZBZNK" "PandoraNet_Hierarchy_DUNEFD_HD_TT_Edge_v014_15_00.pt"
  fi

  if [[ "$2" == "atmos" ]]
  then
    download "2oPV4CspjWR1mHi" "PandoraNet_Vertex_DUNEFD_HD_Atmos_1_U_v04_03_00.pt"
    download "Ok8axRtsxcW1TDj" "PandoraNet_Vertex_DUNEFD_HD_Atmos_1_V_v04_03_00.pt"
    download "zwBHUyYM0CVUOKx" "PandoraNet_Vertex_DUNEFD_HD_Atmos_1_W_v04_03_00.pt"
    download "nKuPKh6vZwlODFJ" "PandoraNet_Vertex_DUNEFD_HD_Atmos_2_U_v04_03_00.pt"
    download "bM7ZRlCRabeRbOu" "PandoraNet_Vertex_DUNEFD_HD_Atmos_2_V_v04_03_00.pt"
    download "QNZfK3YdG2Hebxn" "PandoraNet_Vertex_DUNEFD_HD_Atmos_2_W_v04_03_00.pt"
  fi
fi

# DUNE ND

if [[ "$1" == "dunend" ]]
then
  download "226RBa8gD0ADBHt" "PandoraNet_Vertex_DUNEND_Accel_1_U_v04_06_00.pt"
  download "5QJeL3YxjYQ6Kk4" "PandoraNet_Vertex_DUNEND_Accel_1_V_v04_06_00.pt"
  download "Pyp00cjABgQExjz" "PandoraNet_Vertex_DUNEND_Accel_1_W_v04_06_00.pt"
  download "Gu5AnSMaLboybi5" "PandoraNet_Vertex_DUNEND_Accel_2_U_v04_06_00.pt"
  download "67PwI2wbImXgUeV" "PandoraNet_Vertex_DUNEND_Accel_2_V_v04_06_00.pt"
  download "KxwPe9chjZChUvC" "PandoraNet_Vertex_DUNEND_Accel_2_W_v04_06_00.pt"
fi

cd $MY_TEST_AREA/LArMachineLearningData
