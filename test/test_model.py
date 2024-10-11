import pytest 

class TestFeatureSelection:

    def test_null_values(self, data_results):
        assert data_results.isnull().sum().sum() == 0
    
    def test_zero_values(self, data_results):
        assert data_results.eq(0).sum().sum() == 0
    
    def test_duplicated_values(self, data_results):
        assert data_results.duplicated().sum() == 0