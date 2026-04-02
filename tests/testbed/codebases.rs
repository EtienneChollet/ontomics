use super::expectations::*;
use super::universal;

macro_rules! testbed_tests {
    ($mod_name:ident, $expectations_fn:ident) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn list_concepts() { universal::run_list_concepts(&$expectations_fn()); }

            #[test]
            fn query_concept() { universal::run_query_concept(&$expectations_fn()); }

            #[test]
            fn check_naming() { universal::run_check_naming(&$expectations_fn()); }

            #[test]
            fn suggest_name() { universal::run_suggest_name(&$expectations_fn()); }

            #[test]
            fn list_conventions() { universal::run_list_conventions(&$expectations_fn()); }

            #[test]
            fn describe_symbol() { universal::run_describe_symbol(&$expectations_fn()); }

            #[test]
            fn locate_concept() { universal::run_locate_concept(&$expectations_fn()); }

            #[test]
            fn list_entities() { universal::run_list_entities(&$expectations_fn()); }

            #[test]
            fn export_domain_pack() { universal::run_export_domain_pack(&$expectations_fn()); }

            #[test]
            fn ontology_diff() { universal::run_ontology_diff(&$expectations_fn()); }
        }
    };
    ($mod_name:ident, $expectations_fn:ident, #[ignore]) => {
        mod $mod_name {
            use super::*;

            #[test] #[ignore]
            fn list_concepts() { universal::run_list_concepts(&$expectations_fn()); }

            #[test] #[ignore]
            fn query_concept() { universal::run_query_concept(&$expectations_fn()); }

            #[test] #[ignore]
            fn check_naming() { universal::run_check_naming(&$expectations_fn()); }

            #[test] #[ignore]
            fn suggest_name() { universal::run_suggest_name(&$expectations_fn()); }

            #[test] #[ignore]
            fn list_conventions() { universal::run_list_conventions(&$expectations_fn()); }

            #[test] #[ignore]
            fn describe_symbol() { universal::run_describe_symbol(&$expectations_fn()); }

            #[test] #[ignore]
            fn locate_concept() { universal::run_locate_concept(&$expectations_fn()); }

            #[test] #[ignore]
            fn list_entities() { universal::run_list_entities(&$expectations_fn()); }

            #[test] #[ignore]
            fn export_domain_pack() { universal::run_export_domain_pack(&$expectations_fn()); }

            #[test] #[ignore]
            fn ontology_diff() { universal::run_ontology_diff(&$expectations_fn()); }
        }
    };
}

testbed_tests!(voxelmorph, voxelmorph_expectations);
testbed_tests!(neurite, neurite_expectations);
testbed_tests!(interseg3d, interseg3d_expectations);
testbed_tests!(scribbleprompt, scribbleprompt_expectations);
testbed_tests!(freebrowse, freebrowse_expectations);
testbed_tests!(pylot, pylot_expectations);
testbed_tests!(pytorch, pytorch_expectations, #[ignore]);
testbed_tests!(pandas, pandas_expectations, #[ignore]);
