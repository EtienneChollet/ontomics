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
            fn describe_file() { universal::run_describe_file(&$expectations_fn()); }

            #[test]
            fn locate_concept() { universal::run_locate_concept(&$expectations_fn()); }

            #[test]
            fn list_entities() { universal::run_list_entities(&$expectations_fn()); }

            #[test]
            fn export_domain_pack() { universal::run_export_domain_pack(&$expectations_fn()); }

            #[test]
            fn ontology_diff() { universal::run_ontology_diff(&$expectations_fn()); }

            #[test]
            fn vocabulary_health() { universal::run_vocabulary_health(&$expectations_fn()); }

            #[test]
            fn describe_logic() { universal::run_describe_logic(&$expectations_fn()); }

            #[test]
            fn find_similar_logic() { universal::run_find_similar_logic(&$expectations_fn()); }

            #[test]
            fn compact_context() { universal::run_compact_context(&$expectations_fn()); }

            #[test]
            fn concept_map() { universal::run_concept_map(&$expectations_fn()); }

            #[test]
            fn type_flows() { universal::run_type_flows(&$expectations_fn()); }

            #[test]
            fn trace_type() { universal::run_trace_type(&$expectations_fn()); }

            #[test]
            fn trace_concept() { universal::run_trace_concept(&$expectations_fn()); }
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
            fn describe_file() { universal::run_describe_file(&$expectations_fn()); }

            #[test] #[ignore]
            fn locate_concept() { universal::run_locate_concept(&$expectations_fn()); }

            #[test] #[ignore]
            fn list_entities() { universal::run_list_entities(&$expectations_fn()); }

            #[test] #[ignore]
            fn export_domain_pack() { universal::run_export_domain_pack(&$expectations_fn()); }

            #[test] #[ignore]
            fn ontology_diff() { universal::run_ontology_diff(&$expectations_fn()); }

            #[test] #[ignore]
            fn vocabulary_health() { universal::run_vocabulary_health(&$expectations_fn()); }

            #[test] #[ignore]
            fn describe_logic() { universal::run_describe_logic(&$expectations_fn()); }

            #[test] #[ignore]
            fn find_similar_logic() { universal::run_find_similar_logic(&$expectations_fn()); }

            #[test] #[ignore]
            fn compact_context() { universal::run_compact_context(&$expectations_fn()); }

            #[test] #[ignore]
            fn concept_map() { universal::run_concept_map(&$expectations_fn()); }

            #[test] #[ignore]
            fn type_flows() { universal::run_type_flows(&$expectations_fn()); }

            #[test] #[ignore]
            fn trace_type() { universal::run_trace_type(&$expectations_fn()); }

            #[test] #[ignore]
            fn trace_concept() { universal::run_trace_concept(&$expectations_fn()); }
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
