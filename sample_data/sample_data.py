import datasets
import csv


_DESCRIPTION = """\
Large Movie Review Dataset.
This is a dataset for binary sentiment classification containing substantially \
more data than previous benchmark datasets. We provide a set of 25,000 highly \
polar movie reviews for training, and 25,000 for testing. There is additional \
unlabeled data for use as well.\
"""

_CITATION = """\
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
"""

_DOWNLOAD_URL = "./sample.csv"








class sample_dataConfig(datasets.BuilderConfig):
    """BuilderConfig for sample_data."""

    def __init__(self, **kwargs):
        super().__init__(
            description=f"This is solely for a testing purpose",
            version=datasets.Version("1.0.0",""),
            name = 'sample_data',**kwargs,
        )
        self.date = "20221208"
        self.language = "en"
        


class sample_data(datasets.GeneratorBasedBuilder):
    """Wikipedia dataset."""
    
    VERSION = datasets.Version("1.1.1")
    
    BUILDER_CONFIGS = [sample_dataConfig()]

    DEFAULT_CONFIG_NAME = 'sample_data'
    
    _URL = './sample.csv'
    
    def _info(self):
        return datasets.DatasetInfo(
            description="Testing",
            features=datasets.Features(
                {
                    "first": datasets.Value("float32"),
                    "second": datasets.Value("float32"),
                }
            ),
            # No default supervised_keys.
            supervised_keys=None,
            homepage="https://www.google.com",
            citation="hello world",
        )
    
        
    
    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download(url_or_urls={'dataset':'./sample.csv'})    
        return [datasets.SplitGenerator(  # pylint:disable=g-complex-comprehension
            name=datasets.Split.TRAIN, gen_kwargs={"split": "train", "data_file": downloaded_files["dataset"]}
            )]
    
    
    
    
    def _generate_examples(self, data_file, split):
        with open(data_file) as f:
            reader = csv.reader(f, quoting = csv.QUOTE_NONE)
            feature_names = ['first','second']
            for n, row in enumerate(reader):
                yield n, {feature_names[i]: val for i, val in enumerate(row)}
                
                
                
                