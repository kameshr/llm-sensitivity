"""
Output Classification and Evaluation Module

Implements classification and evaluation of model outputs after parameter corruption,
following the paper's methodology using BERTScore and other metrics.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
try:
    from bert_score import score as bert_score_fn
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    print("Warning: BERTScore not available. Using alternative similarity metrics.")


class OutputEvaluator:
    """
    Evaluates and classifies model outputs after parameter corruption.

    Classification categories (from the paper):
    - Preserved: BERTScore_F1 > 0.87 - meaning relatively unchanged
    - Changed: 0.80 < BERTScore_F1 < 0.87 - intelligible but meaning changed
    - Gibberish: BERTScore_F1 < 0.80 - random jumble of words and symbols
    """

    def __init__(self):
        self.bert_score_model = "microsoft/deberta-xlarge-mnli" if BERT_SCORE_AVAILABLE else None
        self._download_nltk_data()

    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def compute_bert_score(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Compute BERTScore between reference and candidate texts.

        Args:
            reference: Clean/original model output
            candidate: Corrupted model output

        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        if not BERT_SCORE_AVAILABLE:
            # Fallback to simpler similarity metrics
            return self._compute_alternative_similarity(reference, candidate)

        try:
            P, R, F1 = bert_score_fn([candidate], [reference], model_type=self.bert_score_model)
            return {
                'precision': float(P[0]),
                'recall': float(R[0]),
                'f1': float(F1[0])
            }
        except Exception as e:
            print(f"Error computing BERTScore: {e}")
            return self._compute_alternative_similarity(reference, candidate)

    def _compute_alternative_similarity(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Compute alternative similarity metrics when BERTScore is not available.

        Uses word overlap, sentence structure, and token-level similarity.
        """
        # Tokenize texts
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())

        # Remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            ref_tokens = [w for w in ref_tokens if w not in stop_words and w.isalpha()]
            cand_tokens = [w for w in cand_tokens if w not in stop_words and w.isalpha()]
        except:
            ref_tokens = [w for w in ref_tokens if w.isalpha()]
            cand_tokens = [w for w in cand_tokens if w.isalpha()]

        # Calculate word overlap similarity
        ref_set = set(ref_tokens)
        cand_set = set(cand_tokens)

        if len(ref_set) == 0 and len(cand_set) == 0:
            word_overlap = 1.0
        elif len(ref_set) == 0 or len(cand_set) == 0:
            word_overlap = 0.0
        else:
            intersection = len(ref_set.intersection(cand_set))
            union = len(ref_set.union(cand_set))
            word_overlap = intersection / union if union > 0 else 0.0

        # Calculate sequence similarity (Jaccard on bigrams)
        ref_bigrams = set(zip(ref_tokens[:-1], ref_tokens[1:]))
        cand_bigrams = set(zip(cand_tokens[:-1], cand_tokens[1:]))

        if len(ref_bigrams) == 0 and len(cand_bigrams) == 0:
            sequence_sim = 1.0
        elif len(ref_bigrams) == 0 or len(cand_bigrams) == 0:
            sequence_sim = 0.0
        else:
            bigram_intersection = len(ref_bigrams.intersection(cand_bigrams))
            bigram_union = len(ref_bigrams.union(cand_bigrams))
            sequence_sim = bigram_intersection / bigram_union if bigram_union > 0 else 0.0

        # Calculate length similarity
        len_sim = min(len(ref_tokens), len(cand_tokens)) / max(len(ref_tokens), len(cand_tokens), 1)

        # Combine metrics (weighted average approximating BERTScore)
        f1_approx = 0.5 * word_overlap + 0.3 * sequence_sim + 0.2 * len_sim

        return {
            'precision': word_overlap,  # Approximation
            'recall': sequence_sim,     # Approximation
            'f1': f1_approx
        }

    def classify_output(self, reference: str, candidate: str) -> Dict[str, Any]:
        """
        Classify model output based on comparison with reference.

        Args:
            reference: Clean/original model output
            candidate: Corrupted model output

        Returns:
            Dictionary with classification and metrics
        """
        scores = self.compute_bert_score(reference, candidate)
        f1_score = scores['f1']

        # Apply thresholds from the paper
        if f1_score > 0.87:
            classification = 'preserved'
            description = "Meaning relatively unchanged"
        elif 0.80 < f1_score <= 0.87:
            classification = 'changed'
            description = "Intelligible but meaning changed"
        else:
            classification = 'gibberish'
            description = "Random jumble of words and symbols"

        # Additional analysis
        structural_analysis = self._analyze_text_structure(candidate)

        return {
            'classification': classification,
            'description': description,
            'bert_score': scores,
            'f1_score': f1_score,
            'structural_analysis': structural_analysis,
            'texts': {
                'reference': reference,
                'candidate': candidate
            }
        }

    def _analyze_text_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze structural properties of generated text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with structural analysis
        """
        # Basic statistics
        num_chars = len(text)
        num_words = len(text.split())

        # Tokenize for more detailed analysis
        try:
            tokens = word_tokenize(text)
            sentences = sent_tokenize(text)
        except:
            tokens = text.split()
            sentences = text.split('.')

        num_tokens = len(tokens)
        num_sentences = len(sentences)

        # Check for repeated patterns (sign of gibberish)
        token_counts = Counter(tokens)
        most_common_token = token_counts.most_common(1)[0] if token_counts else ('', 0)
        repetition_ratio = most_common_token[1] / num_tokens if num_tokens > 0 else 0

        # Check for unusual punctuation patterns
        punctuation_pattern = re.findall(r'[^\w\s]', text)
        punctuation_density = len(punctuation_pattern) / num_chars if num_chars > 0 else 0

        # Check for excessive commas or special characters (gibberish indicator)
        comma_count = text.count(',')
        comma_density = comma_count / num_chars if num_chars > 0 else 0

        # Average word length
        word_lengths = [len(word) for word in tokens if word.isalpha()]
        avg_word_length = np.mean(word_lengths) if word_lengths else 0

        # Detect potential gibberish patterns
        gibberish_indicators = {
            'high_repetition': repetition_ratio > 0.3,
            'excessive_punctuation': punctuation_density > 0.2,
            'excessive_commas': comma_density > 0.1,
            'very_short_words': avg_word_length < 2.5,
            'no_sentences': num_sentences <= 1 and num_words > 20
        }

        gibberish_score = sum(gibberish_indicators.values()) / len(gibberish_indicators)

        return {
            'num_characters': num_chars,
            'num_words': num_words,
            'num_tokens': num_tokens,
            'num_sentences': num_sentences,
            'avg_word_length': avg_word_length,
            'repetition_ratio': repetition_ratio,
            'punctuation_density': punctuation_density,
            'comma_density': comma_density,
            'gibberish_indicators': gibberish_indicators,
            'gibberish_score': gibberish_score,
            'most_common_token': most_common_token
        }

    def evaluate_experiment_results(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate results from a bit-flip experiment.

        Args:
            experiment_results: Results from BitFlipSimulator.run_bit_flip_experiment

        Returns:
            Dictionary with evaluation summary
        """
        evaluations = []
        classification_counts = {'preserved': 0, 'changed': 0, 'gibberish': 0}
        f1_scores = []

        print(f"Evaluating {len(experiment_results['results'])} experiment trials...")

        for trial_idx, trial in enumerate(experiment_results['results']):
            clean_output = trial['clean_output']
            corrupted_output = trial['corrupted_output']

            evaluation = self.classify_output(clean_output, corrupted_output)
            evaluation.update({
                'trial_index': trial_idx,
                'prompt': trial['prompt'],
                'bit_type': trial['bit_type'],
                'trial_number': trial['trial']
            })

            evaluations.append(evaluation)
            classification_counts[evaluation['classification']] += 1
            f1_scores.append(evaluation['f1_score'])

            if (trial_idx + 1) % 50 == 0:
                print(f"Evaluated {trial_idx + 1} trials...")

        total_trials = len(evaluations)
        classification_percentages = {
            category: (count / total_trials) * 100
            for category, count in classification_counts.items()
        }

        # Analyze by bit type
        bit_type_analysis = {}
        for bit_type in experiment_results['bit_types']:
            bit_type_evals = [e for e in evaluations if e['bit_type'] == bit_type]
            bit_type_counts = {'preserved': 0, 'changed': 0, 'gibberish': 0}

            for eval_result in bit_type_evals:
                bit_type_counts[eval_result['classification']] += 1

            bit_type_total = len(bit_type_evals)
            bit_type_percentages = {
                category: (count / bit_type_total) * 100 if bit_type_total > 0 else 0
                for category, count in bit_type_counts.items()
            }

            bit_type_analysis[bit_type] = {
                'counts': bit_type_counts,
                'percentages': bit_type_percentages,
                'avg_f1': np.mean([e['f1_score'] for e in bit_type_evals]) if bit_type_evals else 0
            }

        return {
            'total_trials': total_trials,
            'overall_classification': {
                'counts': classification_counts,
                'percentages': classification_percentages
            },
            'f1_statistics': {
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores),
                'min': np.min(f1_scores),
                'max': np.max(f1_scores),
                'median': np.median(f1_scores)
            },
            'bit_type_analysis': bit_type_analysis,
            'detailed_evaluations': evaluations
        }

    def create_evaluation_report(self, evaluation_results: Dict[str, Any], save_path: str = None) -> str:
        """
        Create a detailed evaluation report.

        Args:
            evaluation_results: Results from evaluate_experiment_results
            save_path: Optional path to save the report

        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("BIT-FLIP EXPERIMENT EVALUATION REPORT")
        report_lines.append("=" * 80)

        total_trials = evaluation_results['total_trials']
        overall_class = evaluation_results['overall_classification']

        # Overall summary
        report_lines.append(f"\nOVERALL RESULTS ({total_trials} trials):")
        report_lines.append("-" * 40)
        for category, count in overall_class['counts'].items():
            percentage = overall_class['percentages'][category]
            report_lines.append(f"{category.capitalize():<12}: {count:>4} ({percentage:>5.1f}%)")

        # F1 statistics
        f1_stats = evaluation_results['f1_statistics']
        report_lines.append(f"\nBERTScore F1 Statistics:")
        report_lines.append("-" * 40)
        report_lines.append(f"Mean F1:      {f1_stats['mean']:.4f}")
        report_lines.append(f"Std F1:       {f1_stats['std']:.4f}")
        report_lines.append(f"Min F1:       {f1_stats['min']:.4f}")
        report_lines.append(f"Max F1:       {f1_stats['max']:.4f}")
        report_lines.append(f"Median F1:    {f1_stats['median']:.4f}")

        # Bit type analysis
        report_lines.append(f"\nBIT TYPE ANALYSIS:")
        report_lines.append("-" * 40)
        bit_type_analysis = evaluation_results['bit_type_analysis']

        for bit_type, analysis in bit_type_analysis.items():
            report_lines.append(f"\n{bit_type.capitalize()} Bit Corruption:")
            for category, percentage in analysis['percentages'].items():
                count = analysis['counts'][category]
                report_lines.append(f"  {category.capitalize():<12}: {count:>3} ({percentage:>5.1f}%)")
            report_lines.append(f"  Avg F1: {analysis['avg_f1']:.4f}")

        # Example outputs for each category
        report_lines.append(f"\nEXAMPLE OUTPUTS:")
        report_lines.append("-" * 40)

        evaluations = evaluation_results['detailed_evaluations']
        examples_shown = {'preserved': 0, 'changed': 0, 'gibberish': 0}

        for eval_result in evaluations:
            classification = eval_result['classification']
            if examples_shown[classification] < 1:  # Show one example of each
                report_lines.append(f"\n{classification.upper()} Example:")
                report_lines.append(f"Prompt: {eval_result['prompt']}")
                report_lines.append(f"F1 Score: {eval_result['f1_score']:.4f}")
                report_lines.append(f"Clean: {eval_result['texts']['reference'][:100]}...")
                report_lines.append(f"Corrupted: {eval_result['texts']['candidate'][:100]}...")
                examples_shown[classification] += 1

        report_text = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Evaluation report saved to {save_path}")

        return report_text

    def compare_with_paper_results(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare results with the paper's reported findings.

        Paper results for top-3 most sensitive parameters:
        - Preserved: 1.78% (4/225)
        - Changed: 2.67% (6/225)
        - Gibberish: 95.6% (215/225)
        """
        paper_results = {
            'preserved': 1.78,
            'changed': 2.67,
            'gibberish': 95.6
        }

        our_results = evaluation_results['overall_classification']['percentages']

        comparison = {}
        for category in ['preserved', 'changed', 'gibberish']:
            paper_pct = paper_results[category]
            our_pct = our_results[category]
            difference = our_pct - paper_pct

            comparison[category] = {
                'paper_percentage': paper_pct,
                'our_percentage': our_pct,
                'difference': difference,
                'relative_difference': (difference / paper_pct) * 100 if paper_pct > 0 else float('inf')
            }

        return {
            'comparison': comparison,
            'overall_similarity': 1 - (sum(abs(c['difference']) for c in comparison.values()) / 100)
        }


if __name__ == "__main__":
    # Test the output evaluator
    print("Testing output evaluator...")

    evaluator = OutputEvaluator()

    # Test cases
    test_cases = [
        {
            'reference': "The weather today is sunny and warm with clear skies.",
            'candidate': "The weather today is sunny and pleasant with blue skies.",
            'expected': 'preserved'
        },
        {
            'reference': "The weather today is sunny and warm with clear skies.",
            'candidate': "The temperature outside is cold and it's raining heavily.",
            'expected': 'changed'
        },
        {
            'reference': "The weather today is sunny and warm with clear skies.",
            'candidate': ", , , and , , respectively , , . , ,, , or , , are , , ( , , ), , ,",
            'expected': 'gibberish'
        }
    ]

    print("\nTesting classification:")
    for i, test_case in enumerate(test_cases):
        result = evaluator.classify_output(test_case['reference'], test_case['candidate'])
        print(f"Test {i+1}: {result['classification']} (expected: {test_case['expected']})")
        print(f"  F1 Score: {result['f1_score']:.4f}")
        print(f"  Gibberish Score: {result['structural_analysis']['gibberish_score']:.4f}")

    print("\nOutput evaluator test completed!")