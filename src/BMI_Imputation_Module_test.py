from . import BMI_Imputation_Module

def test_BMI_Imputation_Module():
    assert BMI_Imputation_Module.apply("Jane") == "hello Jane"
