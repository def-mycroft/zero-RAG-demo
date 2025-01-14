"""
Class for handling the embedding of text from and providing relevant chunks
based on queries.

This module includes the TextHandler class, which preprocesses text, generates
embeddings using the OpenAI API, indexes them with FAISS, and retrieves relevant
text chunks for user queries. It handles loading, tokenizing, chunking,
embedding, and archiving of the given text data, facilitating efficient and
accurate query responses.

Put more simply, the TextHandler class processes the given text data and
provides features that allow for retrieving only snippets of text that are
potentially relevant to the given query. 

"""
from gpt_general.config import load_config
from zero_rag.imports import * 

import openai
from openai.embeddings_utils import get_embedding
from transformers import GPT2Tokenizer
import faiss

ENGINE = "text-embedding-ada-002"
CONFIG = load_config()
#TOKEN_MAX = CONFIG['rag_tokenmax']
TOKEN_MAX = 1000 # hard coded this for demonstration 
with open(CONFIG['api_key_path'], 'r') as f:
    openai.api_key = f.read().strip()


class TextHandler:
    """
    Handle embedding of text and provide relevant chunks based on queries.

    This class prepares embeddings for text data, generates an index for
    efficient retrieval, and provides relevant text chunks in response
    to queries.

    path_text_input is a directory where the input text data is stored.
    All .txt files at this location will be loaded and used as potential
    context.  This loaded data is stored in the self.pages attribute,
    which is a list of the text of the input data. This text data is
    then tokenized, chunked, and embedded for further processing and
    querying.

    Parameters
    ----------
    pagemax : int, optional
        Maximum number of pages to process (default is np.inf).

    Attributes
    ----------
    config : dict
        Configuration loaded from helpers.
    tokenizer : transformers.GPT2Tokenizer
        Tokenizer for text processing.
    path_text_input : str
        Path to the input text file containing the text data.
    pages : list
        List of pages loaded from the input text file.
    pages_chunked : list
        List of tokenized and chunked pages.
    pages_embedded : list
        List of embeddings for the chunked pages.
    dimension : int
        Dimension of the embeddings.
    index_faiss : faiss.IndexFlatL2
        FAISS index for efficient similarity search.
    page_map : dict
        Mapping of index to chunked pages.
    path_embeddings_archive : str
        Path to the file where embeddings are archived.

    Methods
    -------
    prep_pipeline()
        Full pipeline to prepare for retrieve_relevant_chunks.
    purge_embed_archive()
        Remove the archived embeddings file.
    general_setup()
        Preparatory setup steps.
    process_all_costly()
        Execute costly steps including calls to the OpenAI API for embeddings.
    setup_embeddings()
        Setup embeddings, either by loading from disk or generating new ones.
    generate_index()
        Use FAISS to generate an index from embeddings.
    chunk_pages_tokens(text, max_tokens=1000)
        Tokenize and chunk text, return list of text chunks.
    sample_pages(pages, n)
        Sample a subset of pages for development.
    generate_embeddings(pages)
        Generate embeddings for text using the OpenAI API.
    retrieve_relevant_chunks(query, k=5)
        Get chunks that are relevant to a query.
    """

    def __init__(self, pagemax=np.inf, path_text_input='',
                 prompt_embeddings=False):
        if not path_text_input:
            raise Exception('must have path to json text data. ')
        self.path_text_input = path_text_input
        self.path_project_dir = dirname(path_text_input)
        self.pagemax = pagemax
        self.prompt_embeddings = prompt_embeddings
        self.path_embeddings_archive = \
            join(self.path_project_dir, 'embeddings.pkl')
        self.prep_pipeline()

    def prep_pipeline(self):
        """Full pipeline to prepare for retrieve_relevant_chunks"""
        self.general_setup()
        self.process_all_costly()

    def purge_embed_archive(self):
        os.system(f"rm '{self.path_embeddings_archive}'")
        print(f"removed '{self.path_embeddings_archive}'")

    def general_setup(self):
        """Prepatory setup steps"""
        self.config = load_config()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # here, each element of `pages` is an entire .txt file from 
        # the given repository of text. 
        self.pages = []
        for fp in glob(join(self.path_text_input, '*txt')):
            with open(fp, 'r') as f:
                self.pages.append(f.read())

        if self.pagemax < np.inf:
            raise NotImplementedError()

    def process_all_costly(self):
        """Costly steps incl calls to openai api for embeddings"""
        # tokenize and chunk pages 
        self.pages_chunked = []
        for p in self.pages:
            self.pages_chunked += self.chunk_pages_tokens(p)
        self.setup_embeddings()
        self.generate_index()

    def setup_embeddings(self):
        if exists(self.path_embeddings_archive):
            print(f"note that there are archived embeddings. ")
            pe = pd.read_pickle(self.path_embeddings_archive)
            self.pages_embedded = pe
            print(f"loaded '{self.path_embeddings_archive}'")
        else:
            print('do not have embeddings, re-creating (probably desirable). ')
            pe = self.generate_embeddings(self.pages_chunked)
            pd.to_pickle(pe, self.path_embeddings_archive)
            self.pages_embedded = pe
            print(f"wrote '{self.path_embeddings_archive}'")

    def generate_index(self):
        """Use faiss to generate index"""
        self.dimension = len(self.pages_embedded[0])
        self.index_faiss = faiss.IndexFlatL2(self.dimension)
        self.index_faiss.add(np.array(self.pages_embedded))
        self.page_map = {i:self.pages_chunked[i] 
                         for i in range(len(self.pages_chunked))}

    def chunk_pages_tokens(self, text, max_tokens=TOKEN_MAX):
        """Tokenize and chunk text, return text"""
        tokens = self.tokenizer.tokenize(text)
        chunks = []
        while tokens:
            chunk = tokens[:max_tokens]
            chunks.append(self.tokenizer.convert_tokens_to_string(chunk))
            tokens = tokens[max_tokens:]
        return chunks

    def sample_pages(self, pages, n):
        """Sample pages for dev"""
        keys = list(pages.keys())
        keys = pd.Series(keys).sample(n).tolist()
        return {k:v for k,v in pages.items() if k in keys}

    def generate_embeddings(self, pages):
        """Embed text for openai api"""
        return [get_embedding(x, engine=ENGINE) for x in pages]

    def retrieve_relevant_chunks(self, query, n=5):
        """Get chunks that are relevant to query"""
        query_embedding = get_embedding(query, engine=ENGINE)
        distances, indices = \
            self.index_faiss.search(np.array([query_embedding]), n)
        return [self.page_map[i] for i in indices[0]]
