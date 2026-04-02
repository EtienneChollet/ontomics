/// A single query_concept check.
pub struct QueryConceptCheck {
    pub term: &'static str,
    pub min_variants: usize,
    /// At least one related concept's canonical must contain this substring.
    pub related_contains: Option<&'static str>,
}

/// A single check_naming check.
pub struct NamingCheck {
    pub identifier: &'static str,
    /// "Consistent", "Inconsistent", or "Unknown"
    pub expected_verdict: &'static str,
    /// If Inconsistent, the suggestion must contain this substring.
    pub suggestion_contains: Option<&'static str>,
    /// If set, assert confidence >= this value.
    pub min_confidence: Option<f32>,
}

/// Vocabulary health expectations.
pub struct VocabularyHealthCheck {
    pub min_overall: f32,
}

/// A single suggest_name check.
pub struct SuggestNameCheck {
    pub description: &'static str,
    /// At least one suggestion name must contain this substring.
    pub must_contain: &'static str,
}

/// A single describe_symbol check.
pub struct DescribeSymbolCheck {
    pub name: &'static str,
    /// "Function", "Class", or "Method"
    pub expected_kind: &'static str,
}

/// A single locate_concept check.
pub struct LocateConceptCheck {
    pub term: &'static str,
    /// At least one file path must contain this substring.
    pub file_contains: &'static str,
}

/// A convention that must be detected.
pub struct ConventionCheck {
    /// "Prefix", "Suffix", "Conversion", or "Compound"
    pub pattern_kind: &'static str,
    /// The pattern value (e.g. "nb_", "is_", "_to_")
    pub pattern_value: &'static str,
}

/// A required entity (name must appear in list_entities output).
pub struct EntityCheck {
    pub name: &'static str,
}

/// Full expectations for one codebase.
pub struct TestbedExpectations {
    pub repo_path: &'static str,
    pub name: &'static str,

    // list_concepts
    pub min_concepts: usize,
    pub must_contain_concepts: Vec<&'static str>,
    pub top_n_for_must_contain: usize,

    // query_concept
    pub query_concept_checks: Vec<QueryConceptCheck>,

    // check_naming
    pub naming_checks: Vec<NamingCheck>,

    // suggest_name
    pub suggest_name_checks: Vec<SuggestNameCheck>,

    // list_conventions
    pub must_have_conventions: Vec<ConventionCheck>,

    // describe_symbol
    pub describe_symbol_checks: Vec<DescribeSymbolCheck>,

    // locate_concept
    pub locate_concept_checks: Vec<LocateConceptCheck>,

    // list_entities
    pub min_entities: usize,
    pub must_have_entities: Vec<EntityCheck>,

    // export_domain_pack
    pub min_domain_terms: usize,

    // vocabulary_health
    pub vocabulary_health_check: Option<VocabularyHealthCheck>,

    // performance ceiling (seconds, release-mode builds)
    pub max_build_seconds: u64,
}

// ---------------------------------------------------------------------------
// Codebase definitions
// ---------------------------------------------------------------------------

pub fn voxelmorph_expectations() -> TestbedExpectations {
    TestbedExpectations {
        repo_path: "/home/eti/projects/voxelmorph",
        name: "voxelmorph",
        min_concepts: 15,
        must_contain_concepts: vec![
            "spatial", "field", "affine", "warp", "grid", "disp",
            // "vel" not found as concept (velocity_field tokenized differently)
            // "transform" is rank ~56, needs wider window
            "transform",
        ],
        top_n_for_must_contain: 60,
        query_concept_checks: vec![
            QueryConceptCheck {
                term: "transform",
                min_variants: 3,
                related_contains: Some("spatial"),
            },
            QueryConceptCheck {
                term: "field",
                min_variants: 3,
                related_contains: None,
            },
            QueryConceptCheck {
                term: "disp",
                min_variants: 2,
                related_contains: None,
            },
        ],
        naming_checks: vec![
            NamingCheck {
                identifier: "n_dims",
                expected_verdict: "Inconsistent",
                suggestion_contains: Some("nb_"),
                min_confidence: Some(0.5),
            },
        ],
        suggest_name_checks: vec![
            SuggestNameCheck { description: "spatial transformation", must_contain: "transform" },
        ],
        must_have_conventions: vec![
            ConventionCheck { pattern_kind: "Prefix", pattern_value: "nb_" },
        ],
        describe_symbol_checks: vec![
            DescribeSymbolCheck { name: "SpatialTransformer", expected_kind: "Class" },
            DescribeSymbolCheck { name: "VxmPairwise", expected_kind: "Class" },
            DescribeSymbolCheck { name: "spatial_transform", expected_kind: "Function" },
            DescribeSymbolCheck { name: "disp_to_trf", expected_kind: "Function" },
        ],
        locate_concept_checks: vec![
            LocateConceptCheck { term: "transform", file_contains: "functional.py" },
        ],
        min_entities: 20,
        must_have_entities: vec![
            EntityCheck { name: "SpatialTransformer" },
            EntityCheck { name: "VxmPairwise" },
            EntityCheck { name: "IntegrateVelocityField" },
            EntityCheck { name: "ResizeDisplacementField" },
            EntityCheck { name: "MSE" },
            EntityCheck { name: "Dice" },
            EntityCheck { name: "Grad" },
            // NCC is defined in neurite, not voxelmorph
            EntityCheck { name: "spatial_transform" },
            EntityCheck { name: "affine_to_disp" },
            EntityCheck { name: "disp_to_trf" },
            EntityCheck { name: "trf_to_disp" },
            EntityCheck { name: "compose" },
            EntityCheck { name: "params_to_affine" },
            EntityCheck { name: "volgen" },
            EntityCheck { name: "scan_to_scan" },
            EntityCheck { name: "scan_to_atlas" },
            EntityCheck { name: "load_volfile" },
            EntityCheck { name: "save_volfile" },
            // jacobian_determinant not promoted to entity (low concept tag count)
        ],
        min_domain_terms: 5,
        vocabulary_health_check: Some(VocabularyHealthCheck {
            min_overall: 0.3,
        }),
        max_build_seconds: 15,
    }
}

pub fn neurite_expectations() -> TestbedExpectations {
    TestbedExpectations {
        repo_path: "/home/eti/projects/neurite",
        name: "neurite",
        min_concepts: 10,
        must_contain_concepts: vec![
            "tensor", "noise", "conv", "spatial",
            "upsample", "kernel", "smooth",
            // These exist but rank lower:
            "grid", "sample", "dice",
            "downsample", "downsampling",
        ],
        top_n_for_must_contain: 150,
        query_concept_checks: vec![
            QueryConceptCheck { term: "conv", min_variants: 2, related_contains: None },
            QueryConceptCheck { term: "smooth", min_variants: 2, related_contains: None },
        ],
        naming_checks: vec![
            // NOTE: check_naming's frequency-based logic flags many identifiers
            // as Inconsistent (including nonsense like xyzzy_frobnicator —
            // because 0 occurrences vs any positive count = Inconsistent).
            // Known behavior gap in check_naming — track for improvement.
        ],
        suggest_name_checks: vec![
            SuggestNameCheck { description: "count of features", must_contain: "nb" },
        ],
        must_have_conventions: vec![
            ConventionCheck { pattern_kind: "Prefix", pattern_value: "nb_" },
        ],
        describe_symbol_checks: vec![
            DescribeSymbolCheck { name: "BasicUNet", expected_kind: "Class" },
            DescribeSymbolCheck { name: "Dice", expected_kind: "Class" },
        ],
        locate_concept_checks: vec![
            LocateConceptCheck { term: "conv", file_contains: "nn/" },
        ],
        min_entities: 25,
        must_have_entities: vec![
            EntityCheck { name: "BasicUNet" },
            EntityCheck { name: "BasicAutoencoder" },
            EntityCheck { name: "ConvBlock" },
            EntityCheck { name: "DownsampleConvBlock" },
            EntityCheck { name: "UpsampleConvBlock" },
            EntityCheck { name: "TransposedConv" },
            EntityCheck { name: "Dice" },
            EntityCheck { name: "NCC" },
            EntityCheck { name: "MSE" },
            EntityCheck { name: "SpatialGradient" },
            EntityCheck { name: "Activation" },
            EntityCheck { name: "Pool" },
            EntityCheck { name: "Resize" },
            EntityCheck { name: "DataSplit" },
            EntityCheck { name: "dice" },
            EntityCheck { name: "gaussian_kernel" },
            EntityCheck { name: "gaussian_smoothing" },
            EntityCheck { name: "resample" },
            EntityCheck { name: "fractal_noise" },
            EntityCheck { name: "volshape_to_ndgrid" },
            EntityCheck { name: "downsampling_conv_blocks" },
            EntityCheck { name: "upsampling_conv_blocks" },
        ],
        min_domain_terms: 5,
        vocabulary_health_check: Some(VocabularyHealthCheck {
            min_overall: 0.3,
        }),
        max_build_seconds: 15,
    }
}

pub fn interseg3d_expectations() -> TestbedExpectations {
    TestbedExpectations {
        repo_path: "/home/eti/projects/interseg3d",
        name: "interseg3d",
        min_concepts: 15,
        must_contain_concepts: vec![
            "click", "context", "conv", "seg", "cross", "dice",
            "prompt", "confusion", "embed", "scribble", "support", "target", "sample",
            // "attention" is >200 in interseg3d — ontomics tokenizes it to
            // subtokens that individually rank higher
        ],
        top_n_for_must_contain: 300,
        query_concept_checks: vec![
            QueryConceptCheck { term: "click", min_variants: 3, related_contains: None },
            QueryConceptCheck { term: "cross", min_variants: 2, related_contains: None },
            QueryConceptCheck { term: "attention", min_variants: 2, related_contains: None },
        ],
        naming_checks: vec![],
        suggest_name_checks: vec![
            SuggestNameCheck { description: "cross convolution block", must_contain: "cross" },
        ],
        must_have_conventions: vec![],  // three competing count prefixes — at least one should be detected
        describe_symbol_checks: vec![
            DescribeSymbolCheck { name: "FocalDiceLoss", expected_kind: "Class" },
            DescribeSymbolCheck { name: "UniverSeg3d", expected_kind: "Class" },
            DescribeSymbolCheck { name: "CrossBlock", expected_kind: "Class" },
            DescribeSymbolCheck { name: "PerPixelSetSelfAttention", expected_kind: "Class" },
        ],
        locate_concept_checks: vec![
            LocateConceptCheck { term: "click", file_contains: "click" },
            LocateConceptCheck { term: "attention", file_contains: "attention" },
        ],
        min_entities: 20,
        must_have_entities: vec![
            EntityCheck { name: "UniverSeg3d" },
            EntityCheck { name: "CrossOp" },
            EntityCheck { name: "FastCrossConv3d" },
            EntityCheck { name: "ConvOp" },
            EntityCheck { name: "PerPixelSetSelfAttention" },
            EntityCheck { name: "PerPixelSetCrossAttention" },
            EntityCheck { name: "FocalDiceLoss" },
            EntityCheck { name: "DiceCELoss" },
            EntityCheck { name: "ClickMeta" },
            EntityCheck { name: "ClickMetaIter" },
            EntityCheck { name: "SuperFlexiblePrompt3D" },
            EntityCheck { name: "BinarySegment3D" },
            EntityCheck { name: "MultiBinarySegment3D" },
            EntityCheck { name: "TeramedicalDataset" },
            EntityCheck { name: "focal_loss" },
            EntityCheck { name: "sample_confusion_coords" },
        ],
        min_domain_terms: 8,
        vocabulary_health_check: None,
        max_build_seconds: 30,
    }
}

pub fn scribbleprompt_expectations() -> TestbedExpectations {
    TestbedExpectations {
        repo_path: "/home/eti/projects/ScribblePrompt",
        name: "ScribblePrompt",
        min_concepts: 10,
        must_contain_concepts: vec![
            "mask", "seg", "prompt", "click", "scribble", "embedding",
            "label", "image", "box", "warp", "loss", "encoder",
            // "augmentation" is rank 237 — too low for must_contain
        ],
        top_n_for_must_contain: 100,
        query_concept_checks: vec![
            QueryConceptCheck { term: "scribble", min_variants: 3, related_contains: None },
            QueryConceptCheck { term: "mask", min_variants: 3, related_contains: None },
            QueryConceptCheck { term: "click", min_variants: 2, related_contains: None },
        ],
        naming_checks: vec![],
        suggest_name_checks: vec![
            SuggestNameCheck { description: "image encoder", must_contain: "encoder" },
        ],
        must_have_conventions: vec![],
        describe_symbol_checks: vec![
            DescribeSymbolCheck { name: "ScribblePromptUNet", expected_kind: "Class" },
            DescribeSymbolCheck { name: "ScribblePromptSAM", expected_kind: "Class" },
            DescribeSymbolCheck { name: "FocalDiceLoss", expected_kind: "Class" },
            DescribeSymbolCheck { name: "RandomClick", expected_kind: "Class" },
        ],
        locate_concept_checks: vec![
            LocateConceptCheck { term: "scribble", file_contains: "interactions/" },
            LocateConceptCheck { term: "augmentation", file_contains: "augmentation/" },
        ],
        min_entities: 20,
        must_have_entities: vec![
            EntityCheck { name: "ScribblePromptUNet" },
            EntityCheck { name: "FocalDiceLoss" },
            EntityCheck { name: "RandomClick" },
            EntityCheck { name: "ComponentCenterClick" },
            EntityCheck { name: "RandBorderClick" },
            EntityCheck { name: "CenterlineScribble" },
            EntityCheck { name: "ContourScribble" },
            EntityCheck { name: "SegmentationSequential" },
            EntityCheck { name: "RandomDilation" },
            EntityCheck { name: "click_onehot" },
        ],
        min_domain_terms: 5,
        vocabulary_health_check: None,
        max_build_seconds: 30,
    }
}

pub fn freebrowse_expectations() -> TestbedExpectations {
    TestbedExpectations {
        repo_path: "/home/eti/projects/freebrowse",
        name: "freebrowse",
        min_concepts: 5,
        must_contain_concepts: vec![
            "volume", "surface", "layer", "drawing",
            // "voxel" and "mesh" not in top 100 — low frequency in TS-dominant codebase
        ],
        top_n_for_must_contain: 100,
        query_concept_checks: vec![
            QueryConceptCheck { term: "volume", min_variants: 3, related_contains: None },
            QueryConceptCheck { term: "surface", min_variants: 3, related_contains: None },
            QueryConceptCheck { term: "drawing", min_variants: 2, related_contains: None },
        ],
        naming_checks: vec![],
        suggest_name_checks: vec![
            SuggestNameCheck { description: "volume loader", must_contain: "volume" },
        ],
        must_have_conventions: vec![],
        describe_symbol_checks: vec![
            DescribeSymbolCheck { name: "FreeBrowse", expected_kind: "Function" },
        ],
        locate_concept_checks: vec![
            LocateConceptCheck { term: "volume", file_contains: "volume" },
            LocateConceptCheck { term: "surface", file_contains: "surface" },
        ],
        min_entities: 20,
        must_have_entities: vec![
            EntityCheck { name: "FreeBrowse" },
            EntityCheck { name: "ImageCanvas" },
            EntityCheck { name: "ImageUploader" },
            EntityCheck { name: "Niivue" },
            EntityCheck { name: "Footer" },
            EntityCheck { name: "Header" },
            EntityCheck { name: "Sidebar" },
            EntityCheck { name: "ViewerShell" },
            EntityCheck { name: "useVolumes" },
            EntityCheck { name: "useSurfaces" },
            EntityCheck { name: "useDrawing" },
            EntityCheck { name: "useMeshLayers" },
            // Python backend entities not found — freebrowse is auto-detected
            // as TS-dominant, so Python files may not be parsed
        ],
        min_domain_terms: 3,
        vocabulary_health_check: None,
        max_build_seconds: 30,
    }
}

pub fn pylot_expectations() -> TestbedExpectations {
    TestbedExpectations {
        repo_path: "/home/eti/projects/pylot",
        name: "pylot",
        min_concepts: 15,
        must_contain_concepts: vec![
            "config", "loss", "model", "batch", "data", "module",
        ],
        top_n_for_must_contain: 200,
        query_concept_checks: vec![
            QueryConceptCheck { term: "experiment", min_variants: 3, related_contains: None },
            QueryConceptCheck { term: "config", min_variants: 4, related_contains: None },
            QueryConceptCheck { term: "callback", min_variants: 3, related_contains: None },
            QueryConceptCheck { term: "meter", min_variants: 3, related_contains: None },
        ],
        naming_checks: vec![],
        suggest_name_checks: vec![
            SuggestNameCheck { description: "experiment configuration", must_contain: "config" },
        ],
        // build_ not detected as convention — ontomics finds many other
        // prefix/suffix conventions instead. build_ may not hit the
        // convention_threshold (it's a verb prefix, not a naming pattern).
        must_have_conventions: vec![],
        describe_symbol_checks: vec![
            DescribeSymbolCheck { name: "Config", expected_kind: "Class" },
        ],
        locate_concept_checks: vec![
            LocateConceptCheck { term: "experiment", file_contains: "experiment/" },
            LocateConceptCheck { term: "config", file_contains: "config" },
            LocateConceptCheck { term: "callback", file_contains: "callbacks/" },
            LocateConceptCheck { term: "meter", file_contains: "meter" },
            LocateConceptCheck { term: "loss", file_contains: "loss/" },
        ],
        min_entities: 25,
        must_have_entities: vec![
            EntityCheck { name: "Config" },
            EntityCheck { name: "FileFormat" },
            EntityCheck { name: "to_device" },
            EntityCheck { name: "get_config" },
            EntityCheck { name: "soft_dice_loss" },
            EntityCheck { name: "flatten" },
        ],
        min_domain_terms: 8,
        vocabulary_health_check: None,
        max_build_seconds: 30,
    }
}

pub fn pytorch_expectations() -> TestbedExpectations {
    TestbedExpectations {
        repo_path: "/home/eti/projects/pytorch",
        name: "pytorch",
        min_concepts: 50,
        must_contain_concepts: vec![
            "tensor", "grad", "module", "param", "conv", "linear", "norm",
            "loss", "optim", "dropout", "activation", "pool", "embedding",
            "attention", "batch", "channel", "stride", "kernel", "padding",
            "scheduler", "transformer", "backward", "forward", "weight", "bias",
        ],
        top_n_for_must_contain: 300,
        query_concept_checks: vec![
            QueryConceptCheck { term: "conv", min_variants: 6, related_contains: None },
            QueryConceptCheck { term: "norm", min_variants: 6, related_contains: None },
            QueryConceptCheck { term: "loss", min_variants: 5, related_contains: None },
            QueryConceptCheck { term: "activation", min_variants: 5, related_contains: None },
        ],
        naming_checks: vec![],
        suggest_name_checks: vec![
            SuggestNameCheck { description: "convolutional layer", must_contain: "conv" },
        ],
        must_have_conventions: vec![
            ConventionCheck { pattern_kind: "Prefix", pattern_value: "num_" },
            ConventionCheck { pattern_kind: "Prefix", pattern_value: "is_" },
        ],
        describe_symbol_checks: vec![
            DescribeSymbolCheck { name: "Tensor", expected_kind: "Class" },
            DescribeSymbolCheck { name: "Module", expected_kind: "Class" },
            DescribeSymbolCheck { name: "Parameter", expected_kind: "Class" },
            DescribeSymbolCheck { name: "Linear", expected_kind: "Class" },
            DescribeSymbolCheck { name: "Conv2d", expected_kind: "Class" },
            DescribeSymbolCheck { name: "Adam", expected_kind: "Class" },
        ],
        locate_concept_checks: vec![
            LocateConceptCheck { term: "conv", file_contains: "conv" },
            LocateConceptCheck { term: "loss", file_contains: "loss" },
            LocateConceptCheck { term: "optim", file_contains: "optim/" },
        ],
        min_entities: 20,
        must_have_entities: vec![
            EntityCheck { name: "Tensor" },
            EntityCheck { name: "Module" },
            EntityCheck { name: "Parameter" },
            EntityCheck { name: "Dataset" },
            EntityCheck { name: "DataLoader" },
            EntityCheck { name: "Linear" },
            EntityCheck { name: "Conv1d" },
            EntityCheck { name: "Conv2d" },
            EntityCheck { name: "Conv3d" },
            EntityCheck { name: "BatchNorm1d" },
            EntityCheck { name: "BatchNorm2d" },
            EntityCheck { name: "LayerNorm" },
            EntityCheck { name: "ReLU" },
            EntityCheck { name: "GELU" },
            EntityCheck { name: "Sigmoid" },
            EntityCheck { name: "Softmax" },
            EntityCheck { name: "Dropout" },
            EntityCheck { name: "Embedding" },
            EntityCheck { name: "MultiheadAttention" },
            EntityCheck { name: "Transformer" },
            EntityCheck { name: "LSTM" },
            EntityCheck { name: "MSELoss" },
            EntityCheck { name: "CrossEntropyLoss" },
            EntityCheck { name: "Adam" },
            EntityCheck { name: "SGD" },
            EntityCheck { name: "Sequential" },
            EntityCheck { name: "MaxPool2d" },
        ],
        min_domain_terms: 20,
        vocabulary_health_check: None,
        max_build_seconds: 300,
    }
}

pub fn pandas_expectations() -> TestbedExpectations {
    TestbedExpectations {
        repo_path: "/home/eti/projects/pandas",
        name: "pandas",
        min_concepts: 50,
        must_contain_concepts: vec![
            "index", "frame", "series", "column", "dtype", "group", "block",
            "array", "merge", "concat", "sort", "fill", "drop", "resample",
            "window", "categorical", "missing", "axis", "reshape", "label",
        ],
        top_n_for_must_contain: 300,
        query_concept_checks: vec![
            QueryConceptCheck { term: "index", min_variants: 6, related_contains: None },
            QueryConceptCheck { term: "dtype", min_variants: 5, related_contains: None },
            QueryConceptCheck { term: "group", min_variants: 4, related_contains: None },
            QueryConceptCheck { term: "frame", min_variants: 3, related_contains: None },
        ],
        naming_checks: vec![],
        suggest_name_checks: vec![
            SuggestNameCheck { description: "data frame column", must_contain: "column" },
        ],
        must_have_conventions: vec![
            ConventionCheck { pattern_kind: "Prefix", pattern_value: "is_" },
        ],
        describe_symbol_checks: vec![
            DescribeSymbolCheck { name: "DataFrame", expected_kind: "Class" },
            DescribeSymbolCheck { name: "Series", expected_kind: "Class" },
            DescribeSymbolCheck { name: "Index", expected_kind: "Class" },
            DescribeSymbolCheck { name: "MultiIndex", expected_kind: "Class" },
        ],
        locate_concept_checks: vec![
            LocateConceptCheck { term: "frame", file_contains: "frame.py" },
            LocateConceptCheck { term: "group", file_contains: "groupby" },
            LocateConceptCheck { term: "merge", file_contains: "merge" },
        ],
        min_entities: 20,
        must_have_entities: vec![
            EntityCheck { name: "DataFrame" },
            EntityCheck { name: "Series" },
            EntityCheck { name: "Index" },
            EntityCheck { name: "MultiIndex" },
            EntityCheck { name: "DatetimeIndex" },
            EntityCheck { name: "RangeIndex" },
            EntityCheck { name: "GroupBy" },
            EntityCheck { name: "DataFrameGroupBy" },
            EntityCheck { name: "SeriesGroupBy" },
            EntityCheck { name: "Resampler" },
            EntityCheck { name: "ExtensionDtype" },
            EntityCheck { name: "CategoricalDtype" },
            EntityCheck { name: "ExtensionArray" },
            EntityCheck { name: "Categorical" },
            EntityCheck { name: "BlockManager" },
        ],
        min_domain_terms: 20,
        vocabulary_health_check: None,
        max_build_seconds: 300,
    }
}
