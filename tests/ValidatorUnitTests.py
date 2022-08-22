import unittest

from ml_model_serving.Validator import Validator


class ValidatorTestCase(unittest.TestCase):
    def test_validator_validate_prediction_id_when_empty_then_false(self):

        expectedResult = False
        validator = Validator()

        actualResult = validator.validate_prediction_id("")

        self.assertEqual(expectedResult, actualResult)

    def test_validator_validate_prediction_id_when_none_then_false(self):

        expectedResult = False
        validator = Validator()

        actualResult = validator.validate_prediction_id(None)

        self.assertEqual(expectedResult, actualResult)


if __name__ == '__main__':
    unittest.main()
