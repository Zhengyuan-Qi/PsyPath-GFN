"""Data-free input and inference-interface safety checks."""
import ast
import json
import tempfile
from pathlib import Path
import unittest
from online_pipeline import read_input

class OnlineTests(unittest.TestCase):
    def test_no_labels_or_scores_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json'
            base={'dataset':'daic','dataset_root':d,'participant_id':'123','task':'daic_depression'}
            for key in ['label','score','phq8','cache']:
                p.write_text(json.dumps(dict(base,**{key:1})))
                with self.assertRaises(ValueError): read_input(p)
    def test_release_task_must_match(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json'
            p.write_text(json.dumps({'dataset':'daic','dataset_root':d,'participant_id':'123','task':'edaic_ptsd'}))
            with self.assertRaises(ValueError): read_input(p)
    def test_head_does_not_require_clinical_targets(self):
        tree=ast.parse((Path(__file__).parent/'online_pipeline.py').read_text())
        fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='predict_views')
        strings={n.value for n in ast.walk(fun) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertNotIn('label',strings)
        self.assertNotIn('score',strings)
    def test_valid_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json'
            p.write_text(json.dumps({'dataset':'daic','dataset_root':d,'participant_id':'123','task':'daic_depression'}))
            self.assertEqual(read_input(p)['participant_id'],'123')

if __name__=='__main__': unittest.main()
