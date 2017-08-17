from unittest import TestCase
from .learned import LearnedModelClassifier


class TestLearnedModelClassifier(TestCase):
    def test__strip_remove_non_alpha(self):
        self.assertTrue(isinstance(LearnedModelClassifier._strip_remove_non_alpha('string'), str))

    def test__stem_words_lancaster(self):
        self.assertTrue(isinstance(LearnedModelClassifier._stem_words_lancaster('string string string'), str))
