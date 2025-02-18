void makeTrainingTrees()
{
    TTree *outputPrimaryTrackTree = new TTree("PrimaryTrackTree_TRAIN", "PrimaryTrackTree_TRAIN");
    TTree *outputPrimaryShowerTree = new TTree("PrimaryShowerTree_TRAIN", "PrimaryShowerTree_TRAIN");
    TTree *outputLaterTierTrackTrackTree = new TTree("LaterTierTrackTrackTree_TRAIN", "LaterTierTrackTrackTree_TRAIN");
    TTree *outputLaterTierTrackShowerTree = new TTree("LaterTierTrackShowerTree_TRAIN", "LaterTierTrackShowerTree_TRAIN");

    outputPrimaryTrackTree->SetAutoSave(0);
    outputPrimaryShowerTree->SetAutoSave(0);
    outputLaterTierTrackTrackTree->SetAutoSave(0);
    outputLaterTierTrackShowerTree->SetAutoSave(0);

    outputPrimaryTrackTree->SetDirectory(0);
    outputPrimaryShowerTree->SetDirectory(0);
    outputLaterTierTrackTrackTree->SetDirectory(0);
    outputLaterTierTrackShowerTree->SetDirectory(0);

    ///////////////////////////////////

    int nFiles = 1;

    for (int iFile = 1; iFile < 2; iFile++)
    {
        std::string fileName = "hierarchy_" + to_string(iFile) + ".root";

        std::cout << "fileName: " << fileName << std::endl;

        TFile *inputFile = new TFile(fileName.c_str(), "READ");
        TTree *inputPrimaryTrackTree = (TTree*)inputFile->Get("PrimaryTrackTree");
        TTree *inputPrimaryShowerTree = (TTree*)inputFile->Get("PrimaryShowerTree");
        TTree *inputLaterTierTrackTrackTree = (TTree*)inputFile->Get("LaterTierTrackTrackTree");
        TTree *inputLaterTierTrackShowerTree = (TTree*)inputFile->Get("LaterTierTrackShowerTree");

        // Do primary trees
        for (bool isTrack : {true, false})
        {
            TTree * inputTree(isTrack ? inputPrimaryTrackTree : inputPrimaryShowerTree);
            TTree * outputTree(isTrack ? outputPrimaryTrackTree : outputPrimaryShowerTree);

            int isTrainingLink;
            int isTrueLink;
            int isOrientationCorrect;
            float nSpacepoints;
            float nuSeparation;
            float vertexRegionNHits;
            float vertexRegionNParticles;
            float dca;
            float connectionExtrapDistance;
            float isPOIClosestToNu;
            float parentConnectionDistance;
            float childConnectionDistance;

            inputTree->SetBranchAddress("IsTrainingLink", &isTrainingLink);
            inputTree->SetBranchAddress("IsTrueLink", &isTrueLink);
            inputTree->SetBranchAddress("IsOrientationCorrect", &isOrientationCorrect);
            inputTree->SetBranchAddress("NSpacepoints", &nSpacepoints);
            inputTree->SetBranchAddress("NuSeparation", &nuSeparation);
            inputTree->SetBranchAddress("VertexRegionNHits", &vertexRegionNHits);
            inputTree->SetBranchAddress("VertexRegionNParticles", &vertexRegionNParticles);
            inputTree->SetBranchAddress("DCA", &dca);
            inputTree->SetBranchAddress("ConnectionExtrapDistance", &connectionExtrapDistance);
            inputTree->SetBranchAddress("IsPOIClosestToNu", &isPOIClosestToNu);
            inputTree->SetBranchAddress("ParentConnectionDistance", &parentConnectionDistance);
            inputTree->SetBranchAddress("ChildConnectionDistance", &childConnectionDistance);

            outputTree->Branch("IsTrueLink", &isTrueLink);
            outputTree->Branch("IsOrientationCorrect", &isOrientationCorrect);
            outputTree->Branch("NSpacepoints", &nSpacepoints);
            outputTree->Branch("NuSeparation", &nuSeparation);
            outputTree->Branch("VertexRegionNHits", &vertexRegionNHits);
            outputTree->Branch("VertexRegionNParticles", &vertexRegionNParticles);
            outputTree->Branch("DCA", &dca);
            outputTree->Branch("ConnectionExtrapDistance", &connectionExtrapDistance);
            outputTree->Branch("IsPOIClosestToNu", &isPOIClosestToNu);
            outputTree->Branch("ParentConnectionDistance", &parentConnectionDistance);
            outputTree->Branch("ChildConnectionDistance", &childConnectionDistance);

            for (int i = 0; i < inputTree->GetEntries(); ++i)
            {
                inputTree->GetEntry(i);

                if (isTrainingLink)
                    outputTree->Fill();
            }
        }

        // Do later tier trees
        for (bool isTrack : {true, false})
        {
            TTree * inputTree(isTrack ? inputLaterTierTrackTrackTree : inputLaterTierTrackShowerTree);
            TTree * outputTree(isTrack ? outputLaterTierTrackTrackTree : outputLaterTierTrackShowerTree);

            int isTrainingLink;
            int isTrueLink;
            int isOrientationCorrect;
            int childTrueVisibleGeneration;
            float trainingCutL;
            float trainingCutT;
            float parentTrackScore;
            float childTrackScore;
            float parentNSpacepoints;
            float childNSpacepoints;
            float separation3D;
            float parentNuVertexSep;
            float childNuVertexSep;
            float parentEndRegionNHits;
            float parentEndRegionNParticles;
            float parentEndRegionRToWall;
            float vertexSeparation;
            float doesChildConnect;
            float overshootStartDCA;
            float overshootStartL;
            float overshootEndDCA;
            float overshootEndL;
            float childCPDCA;
            float childCPExtrapDistance;
            float childCPLRatio;
            float parentCPNUpstreamHits;
            float parentCPNDownstreamHits;
            float parentCPNHitRatio;
            float parentCPEigenvalueRatio;
            float parentCPOpeningAngle;
            float parentIsPOIClosestToNu;
            float childIsPOIClosestToNu;

            inputTree->SetBranchAddress("IsTrainingLink", &isTrainingLink);
            inputTree->SetBranchAddress("IsTrueLink", &isTrueLink);
            inputTree->SetBranchAddress("IsOrientationCorrect", &isOrientationCorrect);
            inputTree->SetBranchAddress("ChildTrueVisibleGeneration", &childTrueVisibleGeneration);
            inputTree->SetBranchAddress("TrainingCutL", &trainingCutL);
            inputTree->SetBranchAddress("TrainingCutT", &trainingCutT);
            inputTree->SetBranchAddress("ParentTrackScore", &parentTrackScore);
            inputTree->SetBranchAddress("ChildTrackScore", &childTrackScore);
            inputTree->SetBranchAddress("ParentNSpacepoints", &parentNSpacepoints);
            inputTree->SetBranchAddress("ChildNSpacepoints", &childNSpacepoints);
            inputTree->SetBranchAddress("Separation3D", &separation3D);
            inputTree->SetBranchAddress("ParentNuVertexSep", &parentNuVertexSep);
            inputTree->SetBranchAddress("ChildNuVertexSep", &childNuVertexSep);
            inputTree->SetBranchAddress("ParentEndRegionNHits", &parentEndRegionNHits);
            inputTree->SetBranchAddress("ParentEndRegionNParticles", &parentEndRegionNParticles);
            inputTree->SetBranchAddress("ParentEndRegionRToWall", &parentEndRegionRToWall);
            inputTree->SetBranchAddress("VertexSeparation", &vertexSeparation);
            inputTree->SetBranchAddress("DoesChildConnect", &doesChildConnect);
            inputTree->SetBranchAddress("OvershootStartDCA", &overshootStartDCA);
            inputTree->SetBranchAddress("OvershootStartL", &overshootStartL);
            inputTree->SetBranchAddress("OvershootEndDCA", &overshootEndDCA);
            inputTree->SetBranchAddress("OvershootEndL", &overshootEndL);
            inputTree->SetBranchAddress("ChildCPDCA", &childCPDCA);
            inputTree->SetBranchAddress("ChildCPExtrapDistance", &childCPExtrapDistance);
            inputTree->SetBranchAddress("ChildCPLRatio", &childCPLRatio);
            inputTree->SetBranchAddress("ParentCPNUpstreamHits", &parentCPNUpstreamHits);
            inputTree->SetBranchAddress("ParentCPNDownstreamHits", &parentCPNDownstreamHits);
            inputTree->SetBranchAddress("ParentCPNHitRatio", &parentCPNHitRatio);
            inputTree->SetBranchAddress("ParentCPEigenvalueRatio", &parentCPEigenvalueRatio);
            inputTree->SetBranchAddress("ParentCPOpeningAngle", &parentCPOpeningAngle);
            inputTree->SetBranchAddress("ParentIsPOIClosestToNu", &parentIsPOIClosestToNu);
            inputTree->SetBranchAddress("ChildIsPOIClosestToNu", &childIsPOIClosestToNu);

            outputTree->Branch("IsTrueLink", &isTrueLink);
            outputTree->Branch("IsOrientationCorrect", &isOrientationCorrect);
            outputTree->Branch("ChildTrueVisibleGeneration", &childTrueVisibleGeneration);
            outputTree->Branch("TrainingCutL", &trainingCutL);
            outputTree->Branch("TrainingCutT", &trainingCutT);
            outputTree->Branch("ParentTrackScore", &parentTrackScore);
            outputTree->Branch("ChildTrackScore", &childTrackScore);
            outputTree->Branch("ParentNSpacepoints", &parentNSpacepoints);
            outputTree->Branch("ChildNSpacepoints", &childNSpacepoints);
            outputTree->Branch("Separation3D", &separation3D);
            outputTree->Branch("ParentNuVertexSep", &parentNuVertexSep);
            outputTree->Branch("ChildNuVertexSep", &childNuVertexSep);
            outputTree->Branch("ParentEndRegionNHits", &parentEndRegionNHits);
            outputTree->Branch("ParentEndRegionNParticles", &parentEndRegionNParticles);
            outputTree->Branch("ParentEndRegionRToWall", &parentEndRegionRToWall);
            outputTree->Branch("VertexSeparation", &vertexSeparation);
            outputTree->Branch("DoesChildConnect", &doesChildConnect);
            outputTree->Branch("OvershootStartDCA", &overshootStartDCA);
            outputTree->Branch("OvershootStartL", &overshootStartL);
            outputTree->Branch("OvershootEndDCA", &overshootEndDCA);
            outputTree->Branch("OvershootEndL", &overshootEndL);
            outputTree->Branch("ChildCPDCA", &childCPDCA);
            outputTree->Branch("ChildCPExtrapDistance", &childCPExtrapDistance);
            outputTree->Branch("ChildCPLRatio", &childCPLRatio);
            outputTree->Branch("ParentCPNUpstreamHits", &parentCPNUpstreamHits);
            outputTree->Branch("ParentCPNDownstreamHits", &parentCPNDownstreamHits);
            outputTree->Branch("ParentCPNHitRatio", &parentCPNHitRatio);
            outputTree->Branch("ParentCPEigenvalueRatio", &parentCPEigenvalueRatio);
            outputTree->Branch("ParentCPOpeningAngle", &parentCPOpeningAngle);
            outputTree->Branch("ParentIsPOIClosestToNu", &parentIsPOIClosestToNu);
            outputTree->Branch("ChildIsPOIClosestToNu", &childIsPOIClosestToNu);

            for (int i = 0; i < inputTree->GetEntries(); ++i)
            {
                inputTree->GetEntry(i);

                if (isTrainingLink)
                    outputTree->Fill();
            }
        }

        inputFile->Close();
    }

    TFile *outputFile = new TFile("hierarchy_TRAIN.root", "RECREATE");

    outputPrimaryTrackTree->Write();
    outputPrimaryShowerTree->Write();

    outputLaterTierTrackTrackTree->Write();
    outputLaterTierTrackShowerTree->Write();

    outputFile->Close();
}
