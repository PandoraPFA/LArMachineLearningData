void makeTrainingTrees(const std::string inputFileName, const std::string outputFileName)
{
    // Open input file and get input trees
    TFile *inputFile = new TFile(inputFileName.c_str(), "READ");
    TTree *inputPrimaryTrackTree = (TTree*)inputFile->Get("PrimaryTrackTree");
    TTree *inputPrimaryShowerTree = (TTree*)inputFile->Get("PrimaryShowerTree");
    TTree *inputLaterTierTrackTrackTree = (TTree*)inputFile->Get("LaterTierTrackTrackTree");
    TTree *inputLaterTierTrackShowerTree = (TTree*)inputFile->Get("LaterTierTrackShowerTree");
    
    // Setup output trees
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

    // Fill primary trees
    for (bool isTrack : {true, false})
    {
        TTree * inputTree(isTrack ? inputPrimaryTrackTree : inputPrimaryShowerTree);
        TTree * outputTree(isTrack ? outputPrimaryTrackTree : outputPrimaryShowerTree);

        int isTrainingLink = -999;
        int isTrueLink = -999;
        int isOrientationCorrect = -999;
        float nSpacepoints = -999.f;
        float nuSeparation = -999.f;
        float vertexRegionNHits = -999.f;
        float vertexRegionNParticles = -999.f;
        float dca = -999.f;
        float connectionExtrapDistance = -999.f;
        float isPOIClosestToNu = -999.f;
        float parentConnectionDistance = -999.f;
        float childConnectionDistance = -999.f;

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

    // Fill later tier trees
    for (bool isTrack : {true, false})
    {
        TTree * inputTree(isTrack ? inputLaterTierTrackTrackTree : inputLaterTierTrackShowerTree);
        TTree * outputTree(isTrack ? outputLaterTierTrackTrackTree : outputLaterTierTrackShowerTree);

        int isTrainingLink = -999;
        int isTrueLink = -999;
        int isOrientationCorrect = -999;
        int childTrueVisibleGeneration = -999;
        float trainingCutL = -999.f;
        float trainingCutT = -999.f;
        float parentTrackScore = -999.f;
        float childTrackScore = -999.f;
        float parentNSpacepoints = -999.f;
        float childNSpacepoints = -999.f;
        float separation3D = -999.f;
        float parentNuVertexSep = -999.f;
        float childNuVertexSep = -999.f;
        float parentEndRegionNHits = -999.f;
        float parentEndRegionNParticles = -999.f;
        float parentEndRegionRToWall = -999.f;
        float vertexSeparation = -999.f;
        float doesChildConnect = -999.f;
        float overshootStartDCA = -999.f;
        float overshootStartL = -999.f;
        float overshootEndDCA = -999.f;
        float overshootEndL = -999.f;
        float childCPDCA = -999.f;
        float childCPExtrapDistance = -999.f;
        float childCPLRatio = -999.f;
        float parentCPNUpstreamHits = -999.f;
        float parentCPNDownstreamHits = -999.f;
        float parentCPNHitRatio = -999.f;
        float parentCPEigenvalueRatio = -999.f;
        float parentCPOpeningAngle = -999.f;
        float parentIsPOIClosestToNu = -999.f;
        float childIsPOIClosestToNu = -999.f;

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
    
    // Save output trees in output file
    TFile *outputFile = new TFile(outputFileName.c_str(), "RECREATE");
    outputPrimaryTrackTree->Write();
    outputPrimaryShowerTree->Write();
    outputLaterTierTrackTrackTree->Write();
    outputLaterTierTrackShowerTree->Write();
    outputFile->Close();

    // Mopup
    delete inputFile;
    delete outputPrimaryTrackTree;
    delete outputPrimaryShowerTree;
    delete outputLaterTierTrackTrackTree;
    delete outputLaterTierTrackShowerTree;    
    delete outputFile;
}
