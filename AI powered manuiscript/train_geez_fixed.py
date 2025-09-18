import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeEzDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.examples.append((data['input'], data['target']))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        input_text, target_text = self.examples[idx]
        
        # Tokenize inputs
        input_encoding = self.tokenizer(
            "restore Ge'ez: " + input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize targets
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

def train():
    # Configuration
    config = {
        'model_name': 'google/mt5-small',
        'train_file': 'data/processed/small_training_data.jsonl',
        'val_file': 'data/processed/small_test_data.jsonl',
        'output_dir': 'models/geez_mt5_fixed',
        'max_length': 64,
        'batch_size': 8,
        'num_epochs': 10,
        'learning_rate': 3e-4,
        'warmup_steps': 100,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    logger.info(f"Using device: {config['device']}")
    
    # Initialize tokenizer and model
    logger.info("Initializing tokenizer and model...")
    
    # Force download the tokenizer files to avoid cache issues
    tokenizer = MT5Tokenizer.from_pretrained(
        config['model_name'],
        legacy=False,
        use_fast=True,
        force_download=True,
        local_files_only=False
    )
    
    # Add Ge'ez special tokens
    ge_ez_tokens = [
        '፩', '፪', '፫', '፬', '፭', '፮', '፯', '፰', '፱', '፲', '፳', '፴', '፵', '፶', '፷', '፸', '፹', '፺', '፻', '፼',
        '።', '፣', '፤', '፥', '፦', '፧', '፨', '፠', '፡', '፣', '፤', '፥', '፦', '፧', '፨', '᎐', '᎑', '᎒', '᎓', '᎔', '᎕',
        'ሀ', 'ሁ', 'ሂ', 'ሃ', 'ሄ', 'ህ', 'ሆ', 'ለ', 'ሉ', 'ሊ', 'ላ', 'ሌ', 'ል', 'ሎ', 'ሏ', 'ሐ', 'ሑ', 'ሒ', 'ሓ', 'ሔ', 'ሕ', 'ሖ', 'ሗ',
        'መ', 'ሙ', 'ሚ', 'ማ', 'ሜ', 'ም', 'ሞ', 'ሟ', 'ሠ', 'ሡ', 'ሢ', 'ሣ', 'ሤ', 'ሥ', 'ሦ', 'ሧ', 'ረ', 'ሩ', 'ሪ', 'ራ', 'ሬ', 'ር', 'ሮ', 'ሯ',
        'ሰ', 'ሱ', 'ሲ', 'ሳ', 'ሴ', 'ስ', 'ሶ', 'ሷ', 'ሸ', 'ሹ', 'ሺ', 'ሻ', 'ሼ', 'ሽ', 'ሾ', 'ሿ', 'ቀ', 'ቁ', 'ቂ', 'ቃ', 'ቄ', 'ቅ', 'ቆ', 'ቈ',
        'ቊ', 'ቋ', 'ቌ', 'ቍ', 'በ', 'ቡ', 'ቢ', 'ባ', 'ቤ', 'ብ', 'ቦ', 'ቧ', 'ቨ', 'ቩ', 'ቪ', 'ቫ', 'ቬ', 'ቭ', 'ቮ', 'ቯ', 'ተ', 'ቱ', 'ቲ', 'ታ',
        'ቴ', 'ት', 'ቶ', 'ቷ', 'ቸ', 'ቹ', 'ቺ', 'ቻ', 'ቼ', 'ች', 'ቾ', 'ቿ', 'ኀ', 'ኁ', 'ኂ', 'ኃ', 'ኄ', 'ኅ', 'ኆ', 'ኈ', 'ኊ', 'ኋ', 'ኌ', 'ኍ',
        'ነ', 'ኑ', 'ኒ', 'ና', 'ኔ', 'ን', 'ኖ', 'ኗ', 'ኘ', 'ኙ', 'ኚ', 'ኛ', 'ኜ', 'ኝ', 'ኞ', 'ኟ', 'አ', 'ኡ', 'ኢ', 'ኣ', 'ኤ', 'እ', 'ኦ', 'ኧ',
        'ከ', 'ኩ', 'ኪ', 'ካ', 'ኬ', 'ክ', 'ኮ', 'ኰ', 'ኲ', 'ኳ', 'ኴ', 'ኵ', 'ኸ', 'ኹ', 'ኺ', 'ኻ', 'ኼ', 'ኽ', 'ኾ', 'ዀ', 'ዂ', 'ዃ', 'ዄ', 'ዅ',
        'ወ', 'ዉ', 'ዊ', 'ዋ', 'ዌ', 'ው', 'ዎ', 'ዐ', 'ዑ', 'ዒ', 'ዓ', 'ዔ', 'ዕ', 'ዖ', 'ዘ', 'ዙ', 'ዚ', 'ዛ', 'ዜ', 'ዝ', 'ዞ', 'ዟ',
        'ዠ', 'ዡ', 'ዢ', 'ዣ', 'ዤ', 'ዥ', 'ዦ', 'ዧ', 'የ', 'ዩ', 'ዪ', 'ያ', 'ዬ', 'ይ', 'ዮ', 'ደ', 'ዱ', 'ዲ', 'ዳ', 'ዴ', 'ድ', 'ዶ', 'ዷ',
        'ጀ', 'ጁ', 'ጂ', 'ጃ', 'ጄ', 'ጅ', 'ጆ', 'ጇ', 'ገ', 'ጉ', 'ጊ', 'ጋ', 'ጌ', 'ግ', 'ጎ', 'ጐ', 'ጒ', 'ጓ', 'ጔ', 'ጕ', 'ጠ', 'ጡ',
        'ጢ', 'ጣ', 'ጤ', 'ጥ', 'ጦ', 'ጧ', 'ጨ', 'ጩ', 'ጪ', 'ጫ', 'ጬ', 'ጭ', 'ጮ', 'ጯ', 'ጰ', 'ጱ', 'ጲ', 'ጳ', 'ጴ', 'ጵ', 'ጶ', 'ጷ',
        'ጸ', 'ጹ', 'ጺ', 'ጻ', 'ጼ', 'ጽ', 'ጾ', 'ጿ', 'ፀ', 'ፁ', 'ፂ', 'ፃ', 'ፄ', 'ፅ', 'ፆ', 'ፈ', 'ፉ', 'ፊ', 'ፋ', 'ፌ', 'ፍ', 'ፎ', 'ፏ',
        'ፐ', 'ፑ', 'ፒ', 'ፓ', 'ፔ', 'ፕ', 'ፖ', 'ፗ', 'ፘ', 'ፙ', 'ፚ', '፝', '፞', '፟', '፠', '፡', '።', '፣', '፤', '፥', '፦', '፧',
        '፨', '፩', '፪', '፫', '፬', '፭', '፮', '፯', '፰', '፱', '፲', '፳', '፴', '፵', '፶', '፷', '፸', '፹', '፺', '፻', '፼', '᎐',
        '᎑', '᎒', '᎓', '᎔', '᎕', '᎖', '᎗', '᎘', '᎙', 'Ꭰ', 'Ꭱ', 'Ꭲ', 'Ꭳ', 'Ꭴ', 'Ꭵ', 'Ꭶ', 'Ꭷ', 'Ꭸ', 'Ꭹ', 'Ꭺ', 'Ꭻ', 'Ꭼ',
        'Ꭽ', 'Ꭾ', 'Ꭿ', 'Ꮀ', 'Ꮁ', 'Ꮂ', 'Ꮃ', 'Ꮄ', 'Ꮅ', 'Ꮆ', 'Ꮇ', 'Ꮈ', 'Ꮉ', 'Ꮊ', 'Ꮋ', 'Ꮌ', 'Ꮍ', 'Ꮎ', 'Ꮏ', 'Ꮐ', 'Ꮑ', 'Ꮒ',
        'Ꮓ', 'Ꮔ', 'Ꮕ', 'Ꮖ', 'Ꮗ', 'Ꮘ', 'Ꮙ', 'Ꮚ', 'Ꮛ', 'Ꮜ', 'Ꮝ', 'Ꮞ', 'Ꮟ', 'Ꮠ', 'Ꮡ', 'Ꮢ', 'Ꮣ', 'Ꮤ', 'Ꮥ', 'Ꮦ', 'Ꮧ', 'Ꮨ',
        'Ꮩ', 'Ꮪ', 'Ꮫ', 'Ꮬ', 'Ꮭ', 'Ꮮ', 'Ꮯ', 'Ꮰ', 'Ꮱ', 'Ꮲ', 'Ꮳ', 'Ꮴ', 'Ꮵ', 'Ꮶ', 'Ꮷ', 'Ꮸ', 'Ꮹ', 'Ꮺ', 'Ꮻ', 'Ꮼ', 'Ꮽ', 'Ꮾ',
        'Ꮿ', 'Ᏸ', 'Ᏹ', 'Ᏺ', 'Ᏻ', 'Ᏼ', 'Ᏽ', 'ᏸ', 'ᏹ', 'ᏺ', 'ᏻ', 'ᏼ', 'ᏽ', 'ᐁ', 'ᐂ', 'ᐃ', 'ᐄ', 'ᐅ', 'ᐆ', 'ᐇ', 'ᐈ', 'ᐉ',
        'ᐊ', 'ᐋ', 'ᐌ', 'ᐍ', 'ᐎ', 'ᐏ', 'ᐐ', 'ᐑ', 'ᐒ', 'ᐓ', 'ᐔ', 'ᐕ', 'ᐖ', 'ᐗ', 'ᐘ', 'ᐙ', 'ᐚ', 'ᐛ', 'ᐜ', 'ᐝ', 'ᐞ', 'ᐟ',
        'ᐠ', 'ᐡ', 'ᐢ', 'ᐣ', 'ᐤ', 'ᐥ', 'ᐦ', 'ᐧ', 'ᐨ', 'ᐩ', 'ᐪ', 'ᐫ', 'ᐬ', 'ᐭ', 'ᐮ', 'ᐯ', 'ᐰ', 'ᐱ', 'ᐲ', 'ᐳ', 'ᐴ', 'ᐵ',
        'ᐶ', 'ᐷ', 'ᐸ', 'ᐹ', 'ᐺ', 'ᐻ', 'ᐼ', 'ᐽ', 'ᐾ', 'ᐿ', 'ᑀ', 'ᑁ', 'ᑂ', 'ᑃ', 'ᑄ', 'ᑅ', 'ᑆ', 'ᑇ', 'ᑈ', 'ᑉ', 'ᑊ', 'ᑋ',
        'ᑌ', 'ᑍ', 'ᑎ', 'ᑏ', 'ᑐ', 'ᑑ', 'ᑒ', 'ᑓ', 'ᑔ', 'ᑕ', 'ᑖ', 'ᑗ', 'ᑘ', 'ᑙ', 'ᑚ', 'ᑛ', 'ᑜ', 'ᑝ', 'ᑞ', 'ᑟ', 'ᑠ', 'ᑡ',
        'ᑢ', 'ᑣ', 'ᑤ', 'ᑥ', 'ᑦ', 'ᑧ', 'ᑨ', 'ᑩ', 'ᑪ', 'ᑫ', 'ᑬ', 'ᑭ', 'ᑮ', 'ᑯ', 'ᑰ', 'ᑱ', 'ᑲ', 'ᑳ', 'ᑴ', 'ᑵ', 'ᑶ', 'ᑷ',
        'ᑸ', 'ᑹ', 'ᑺ', 'ᑻ', 'ᑼ', 'ᑽ', 'ᑾ', 'ᑿ', 'ᒀ', 'ᒁ', 'ᒂ', 'ᒃ', 'ᒄ', 'ᒅ', 'ᒆ', 'ᒇ', 'ᒈ', 'ᒉ', 'ᒊ', 'ᒋ', 'ᒌ', 'ᒍ',
        'ᒎ', 'ᒏ', 'ᒐ', 'ᒑ', 'ᒒ', 'ᒓ', 'ᒔ', 'ᒕ', 'ᒖ', 'ᒗ', 'ᒘ', 'ᒙ', 'ᒚ', 'ᒛ', 'ᒜ', 'ᒝ', 'ᒞ', 'ᒟ', 'ᒠ', 'ᒡ', 'ᒢ', 'ᒣ',
        'ᒤ', 'ᒥ', 'ᒦ', 'ᒧ', 'ᒨ', 'ᒩ', 'ᒪ', 'ᒫ', 'ᒬ', 'ᒭ', 'ᒮ', 'ᒯ', 'ᒰ', 'ᒱ', 'ᒲ', 'ᒳ', 'ᒴ', 'ᒵ', 'ᒶ', 'ᒷ', 'ᒸ', 'ᒹ',
        'ᒺ', 'ᒻ', 'ᒼ', 'ᒽ', 'ᒾ', 'ᒿ', 'ᓀ', 'ᓁ', 'ᓂ', 'ᓃ', 'ᓄ', 'ᓅ', 'ᓆ', 'ᓇ', 'ᓈ', 'ᓉ', 'ᓊ', 'ᓋ', 'ᓌ', 'ᓍ', 'ᓎ', 'ᓏ',
        'ᓐ', 'ᓑ', 'ᓒ', 'ᓓ', 'ᓔ', 'ᓕ', 'ᓖ', 'ᓗ', 'ᓘ', 'ᓙ', 'ᓚ', 'ᓛ', 'ᓜ', 'ᓝ', 'ᓞ', 'ᓟ', 'ᓠ', 'ᓡ', 'ᓢ', 'ᓣ', 'ᓤ', 'ᓥ',
        'ᓦ', 'ᓧ', 'ᓨ', 'ᓩ', 'ᓪ', 'ᓫ', 'ᓬ', 'ᓭ', 'ᓮ', 'ᓯ', 'ᓰ', 'ᓱ', 'ᓲ', 'ᓳ', 'ᓴ', 'ᓵ', 'ᓶ', 'ᓷ', 'ᓸ', 'ᓹ', 'ᓺ', 'ᓻ',
        'ᓼ', 'ᓽ', 'ᓾ', 'ᓿ', 'ᔀ', 'ᔁ', 'ᔂ', 'ᔃ', 'ᔄ', 'ᔅ', 'ᔆ', 'ᔇ', 'ᔈ', 'ᔉ', 'ᔊ', 'ᔋ', 'ᔌ', 'ᔍ', 'ᔎ', 'ᔏ', 'ᔐ', 'ᔑ',
        'ᔒ', 'ᔓ', 'ᔔ', 'ᔕ', 'ᔖ', 'ᔗ', 'ᔘ', 'ᔙ', 'ᔚ', 'ᔛ', 'ᔜ', 'ᔝ', 'ᔞ', 'ᔟ', 'ᔠ', 'ᔡ', 'ᔢ', 'ᔣ', 'ᔤ', 'ᔥ', 'ᔦ', 'ᔧ',
        'ᔨ', 'ᔩ', 'ᔪ', 'ᔫ', 'ᔬ', 'ᔭ', 'ᔮ', 'ᔯ', 'ᔰ', 'ᔱ', 'ᔲ', 'ᔳ', 'ᔴ', 'ᔵ', 'ᔶ', 'ᔷ', 'ᔸ', 'ᔹ', 'ᔺ', 'ᔻ', 'ᔼ', 'ᔽ',
        'ᔾ', 'ᔿ', 'ᕀ', 'ᕁ', 'ᕂ', 'ᕃ', 'ᕄ', 'ᕅ', 'ᕆ', 'ᕇ', 'ᕈ', 'ᕉ', 'ᕊ', 'ᕋ', 'ᕌ', 'ᕕ', 'ᕖ', 'ᕗ', 'ᕘ', 'ᕙ', 'ᕚ', 'ᕛ',
        'ᕜ', 'ᕝ', 'ᕞ', 'ᕟ', 'ᕠ', 'ᕡ', 'ᕢ', 'ᕣ', 'ᕤ', 'ᕥ', 'ᕦ', 'ᕧ', 'ᕨ', 'ᕩ', 'ᕪ', 'ᕫ', 'ᕬ', 'ᕭ', 'ᕮ', 'ᕯ', 'ᕰ', 'ᕱ',
        'ᕲ', 'ᕳ', 'ᕴ', 'ᕵ', 'ᕶ', 'ᕷ', 'ᕸ', 'ᕹ', 'ᕺ', 'ᕻ', 'ᕼ', 'ᕽ', 'ᕾ', 'ᕿ', 'ᖀ', 'ᖁ', 'ᖂ', 'ᖃ', 'ᖄ', 'ᖅ', 'ᖆ', 'ᖇ',
        'ᖈ', 'ᖉ', 'ᖊ', 'ᖋ', 'ᖌ', 'ᖍ', 'ᖎ', 'ᖏ', 'ᖐ', 'ᖑ', 'ᖒ', 'ᖓ', 'ᖔ', 'ᖕ', 'ᖖ', 'ᖗ', 'ᖘ', 'ᖙ', 'ᖚ', 'ᖛ', 'ᖜ', 'ᖝ',
        'ᖞ', 'ᖟ', 'ᖠ', 'ᖡ', 'ᖢ', 'ᖣ', 'ᖤ', 'ᖥ', 'ᖦ', 'ᖧ', 'ᖨ', 'ᖩ', 'ᖪ', 'ᖫ', 'ᖬ', 'ᖭ', 'ᖮ', 'ᖯ', 'ᖰ', 'ᖱ', 'ᖲ', 'ᖳ',
        'ᖴ', 'ᖵ', 'ᖶ', 'ᖷ', 'ᖸ', 'ᖹ', 'ᖺ', 'ᖻ', 'ᖼ', 'ᖽ', 'ᖾ', 'ᖿ', 'ᗀ', 'ᗁ', 'ᗂ', 'ᗃ', 'ᗄ', 'ᗅ', 'ᗆ', 'ᗇ', 'ᗈ', 'ᗉ',
        'ᗊ', 'ᗋ', 'ᗌ', 'ᗍ', 'ᗎ', 'ᗏ', 'ᗐ', 'ᗑ', 'ᗒ', 'ᗓ', 'ᗔ', 'ᗕ', 'ᗖ', 'ᗗ', 'ᗘ', 'ᗙ', 'ᗚ', 'ᗛ', 'ᗜ', 'ᗝ', 'ᗞ', 'ᗟ',
        'ᗠ', 'ᗡ', 'ᗢ', 'ᗣ', 'ᗤ', 'ᗥ', 'ᗦ', 'ᗧ', 'ᗨ', 'ᗩ', 'ᗪ', 'ᗫ', 'ᗬ', 'ᗭ', 'ᗮ', 'ᗯ', 'ᗰ', 'ᗱ', 'ᗲ', 'ᗳ', 'ᗴ', 'ᗵ',
        'ᗶ', 'ᗷ', 'ᗸ', 'ᗹ', 'ᗺ', 'ᗻ', 'ᗼ', 'ᗽ', 'ᗾ', 'ᗿ', 'ᘀ', 'ᘁ', 'ᘂ', 'ᘃ', 'ᘄ', 'ᘅ', 'ᘆ', 'ᘇ', 'ᘈ', 'ᘉ', 'ᘊ', 'ᘋ',
        'ᘌ', 'ᘍ', 'ᘎ', 'ᘏ', 'ᘐ', 'ᘑ', 'ᘒ', 'ᘓ', 'ᘔ', 'ᘕ', 'ᘖ', 'ᘗ', 'ᘘ', 'ᘙ', 'ᘚ', 'ᘛ', 'ᘜ', 'ᘝ', 'ᘞ', 'ᘟ', 'ᘠ', 'ᘡ',
        'ᘢ', 'ᘣ', 'ᘤ', 'ᘥ', 'ᘦ', 'ᘧ', 'ᘨ', 'ᘩ', 'ᘪ', 'ᘫ', 'ᘬ', 'ᘭ', 'ᘮ', 'ᘯ', 'ᘰ', 'ᘱ', 'ᘲ', 'ᘳ', 'ᘴ', 'ᘵ', 'ᘶ', 'ᘷ',
        'ᘸ', 'ᘹ', 'ᘺ', 'ᘻ', 'ᘼ', 'ᘽ', 'ᘾ', 'ᘿ', 'ᙀ', 'ᙁ', 'ᙂ', 'ᙃ', 'ᙄ', 'ᙅ', 'ᙆ', 'ᙇ', 'ᙈ', 'ᙉ', 'ᙊ', 'ᙋ', 'ᙌ', 'ᙍ',
        'ᙎ', 'ᙏ', 'ᙐ', 'ᙑ', 'ᙒ', 'ᙓ', 'ᙔ', 'ᙕ', 'ᙖ', 'ᙗ', 'ᙘ', 'ᙙ', 'ᙚ', 'ᙛ', 'ᙜ', 'ᙝ', 'ᙞ', 'ᙟ', 'ᙠ', 'ᙡ', 'ᙢ', 'ᙣ',
        'ᙤ', 'ᙥ', 'ᙦ', 'ᙧ', 'ᙨ', 'ᙩ', 'ᙪ', 'ᙫ', 'ᙬ', '᙭', '᙮', 'ᙯ', 'ᙰ', 'ᙱ', 'ᙲ', 'ᙳ', 'ᙴ', 'ᙵ', 'ᙶ', 'ᙷ', 'ᙸ', 'ᙹ',
        'ᙺ', 'ᙻ', 'ᙼ', 'ᙽ', 'ᙾ', 'ᙿ', 'ᚁ', 'ᚂ', 'ᚃ', 'ᚄ', 'ᚅ', 'ᚆ', 'ᚇ', 'ᚈ', 'ᚉ', 'ᚊ', 'ᚋ', 'ᚌ', 'ᚍ', 'ᚎ', 'ᚏ', 'ᚐ',
        'ᚑ', 'ᚒ', 'ᚓ', 'ᚔ', 'ᚕ', 'ᚖ', 'ᚗ', 'ᚘ', 'ᚙ', 'ᚚ', '᚛', '᚜', 'ᚠ', 'ᚡ', 'ᚢ', 'ᚣ', 'ᚤ', 'ᚥ', 'ᚦ', 'ᚧ', 'ᚨ', 'ᚩ',
        'ᚪ', 'ᚫ', 'ᚬ', 'ᚭ', 'ᚮ', 'ᚯ', 'ᚰ', 'ᚱ', 'ᚲ', 'ᚳ', 'ᚴ', 'ᚵ', 'ᚶ', 'ᚷ', 'ᚸ', 'ᚹ', 'ᚺ', 'ᚻ', 'ᚼ', 'ᚽ', 'ᚾ', 'ᚿ',
        'ᛀ', 'ᛁ', 'ᛂ', 'ᛃ', 'ᛄ', 'ᛅ', 'ᛆ', 'ᛇ', 'ᛈ', 'ᛉ', 'ᛊ', 'ᛋ', 'ᛌ', 'ᛍ', 'ᛎ', 'ᛏ', 'ᛐ', 'ᛑ', 'ᛒ', 'ᛓ', 'ᛔ', 'ᛕ',
        'ᛖ', 'ᛗ', 'ᛘ', 'ᛙ', 'ᛚ', 'ᛛ', 'ᛜ', 'ᛝ', 'ᛞ', 'ᛟ', 'ᛠ', 'ᛡ', 'ᛢ', 'ᛣ', 'ᛤ', 'ᛥ', 'ᛦ', 'ᛧ', 'ᛨ', 'ᛩ', 'ᛪ', '᛫',
        '᛬', '᛭', 'ᛮ', 'ᛯ', 'ᛰ', 'ᛱ', 'ᛲ', 'ᛳ', 'ᛴ', 'ᛵ', 'ᛶ', 'ᛷ', 'ᛸ'
    ]
    
    # Add special tokens
    special_tokens = {
        'additional_special_tokens': ['<sep>', '<cls>', '<mask>'] + ge_ez_tokens
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # Initialize model with proper configuration
    model = MT5ForConditionalGeneration.from_pretrained(
        config['model_name'],
        max_length=config['max_length']
    )
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(config['device'])
    
    # Setup training data
    train_dataset = GeEzDataset(config['train_file'], tokenizer, config['max_length'])
    val_dataset = GeEzDataset(config['val_file'], tokenizer, config['max_length'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate']
    )
    
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(config['device'])
            attention_mask = batch['attention_mask'].to(config['device'])
            labels = batch['labels'].to(config['device'])
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Calculate loss
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss for the epoch
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(DataLoader(val_dataset, batch_size=config['batch_size']), desc="Validating"):
                input_ids = batch['input_ids'].to(config['device'])
                attention_mask = batch['attention_mask'].to(config['device'])
                labels = batch['labels'].to(config['device'])
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(DataLoader(val_dataset))
        logger.info(f"Epoch {epoch + 1} - Validation loss: {avg_val_loss:.4f}")
        
        # Save the best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model.save_pretrained(config['output_dir'])
            tokenizer.save_pretrained(config['output_dir'])
            logger.info(f"New best model saved to {config['output_dir']}")
    
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_loss:.4f}")
    
    # Generate sample predictions
    sample_input = "ንጉሥ ፥ ቤተ ፥ ክርስትያን ። ነቢይ"
    input_encoding = tokenizer(
        "restore Ge'ez: " + sample_input,
        max_length=config['max_length'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(config['device'])
    
    generated_ids = model.generate(
        input_ids=input_encoding['input_ids'],
        attention_mask=input_encoding['attention_mask'],
        max_length=64,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        decoder_start_token_id=tokenizer.pad_token_id
    )
    
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    logger.info(f"\nSample prediction:")
    logger.info(f"Input:   {sample_input}")
    logger.info(f"Expected: {sample_input.replace('ንጉሥ', 'መንግሥት')}")
    logger.info(f"Output:  {output}")

if __name__ == "__main__":
    train()
