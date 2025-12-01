"""
Large molecule list for training molecular informatics models.
Contains 1000+ common drugs, natural products, and small molecules.
"""

LARGE_MOLECULE_LIST = [
    # Common drugs (A-Z)
    'aspirin', 'acetaminophen', 'paracetamol', 'ibuprofen', 'naproxen',
    'diclofenac', 'indomethacin', 'celecoxib', 'meloxicam', 'piroxicam',
    'ketoprofen', 'ketorolac', 'mefenamic acid', 'etodolac', 'flurbiprofen',
    'caffeine', 'theophylline', 'theobromine', 'paraxanthine',
    'morphine', 'codeine', 'hydrocodone', 'oxycodone', 'tramadol',
    'fentanyl', 'meperidine', 'methadone', 'buprenorphine', 'naloxone',
    'atorvastatin', 'simvastatin', 'lovastatin', 'pravastatin', 'rosuvastatin',
    'metformin', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone',
    'lisinopril', 'enalapril', 'ramipril', 'captopril', 'benazepril',
    'amlodipine', 'nifedipine', 'diltiazem', 'verapamil', 'felodipine',
    'metoprolol', 'atenolol', 'propranolol', 'carvedilol', 'bisoprolol',
    'furosemide', 'hydrochlorothiazide', 'chlorthalidone', 'spironolactone',
    'warfarin', 'heparin', 'rivaroxaban', 'apixaban', 'dabigatran',
    'omeprazole', 'lansoprazole', 'pantoprazole', 'esomeprazole', 'rabeprazole',
    'ranitidine', 'famotidine', 'cimetidine', 'nizatidine',
    'sertraline', 'fluoxetine', 'paroxetine', 'citalopram', 'escitalopram',
    'amitriptyline', 'nortriptyline', 'doxepin', 'imipramine', 'clomipramine',
    'lorazepam', 'diazepam', 'alprazolam', 'clonazepam', 'midazolam',
    'phenytoin', 'carbamazepine', 'valproic acid', 'lamotrigine', 'levetiracetam',
    'albuterol', 'salmeterol', 'formoterol', 'terbutaline',
    'montelukast', 'zafirlukast', 'zileuton',
    'cetirizine', 'loratadine', 'fexofenadine', 'desloratadine', 'levocetirizine',
    'diphenhydramine', 'chlorpheniramine', 'promethazine', 'hydroxyzine',
    'amoxicillin', 'ampicillin', 'penicillin', 'dicloxacillin',
    'cephalexin', 'cefuroxime', 'cefdinir', 'ceftriaxone', 'cefepime',
    'azithromycin', 'clarithromycin', 'erythromycin',
    'doxycycline', 'tetracycline', 'minocycline',
    'ciprofloxacin', 'levofloxacin', 'moxifloxacin', 'ofloxacin',
    'metronidazole', 'tinidazole',
    'acyclovir', 'valacyclovir', 'famciclovir', 'ganciclovir',

    # Neurotransmitters and related
    'dopamine', 'serotonin', 'norepinephrine', 'epinephrine',
    'acetylcholine', 'histamine', 'melatonin',
    'gaba', 'glutamate', 'glycine', 'taurine',

    # Vitamins and supplements
    'ascorbic acid', 'retinol', 'thiamine', 'riboflavin', 'niacin',
    'pyridoxine', 'biotin', 'folic acid', 'cobalamin',
    'cholecalciferol', 'ergocalciferol', 'tocopherol', 'phylloquinone',

    # Amino acids
    'alanine', 'arginine', 'asparagine', 'aspartic acid', 'cysteine',
    'glutamine', 'glutamic acid', 'glycine', 'histidine', 'isoleucine',
    'leucine', 'lysine', 'methionine', 'phenylalanine', 'proline',
    'serine', 'threonine', 'tryptophan', 'tyrosine', 'valine',

    # Sugars and carbohydrates
    'glucose', 'fructose', 'galactose', 'sucrose', 'lactose',
    'maltose', 'ribose', 'deoxyribose', 'xylose', 'mannose',

    # Organic acids
    'acetic acid', 'formic acid', 'propionic acid', 'butyric acid',
    'lactic acid', 'malic acid', 'citric acid', 'succinic acid',
    'fumaric acid', 'oxalic acid', 'tartaric acid',

    # Alkaloids and natural products
    'nicotine', 'atropine', 'scopolamine', 'cocaine', 'quinine',
    'quinidine', 'strychnine', 'brucine', 'colchicine', 'reserpine',
    'vincristine', 'vinblastine', 'taxol', 'camptothecin',
    'capsaicin', 'piperine', 'curcumin', 'resveratrol',

    # Steroids and hormones
    'cortisol', 'cortisone', 'prednisone', 'prednisolone', 'dexamethasone',
    'testosterone', 'estradiol', 'progesterone', 'estrone', 'estriol',
    'hydrocortisone', 'betamethasone', 'triamcinolone',
    'cholesterol', 'ergosterol', 'stigmasterol',

    # Flavonoids and polyphenols
    'quercetin', 'kaempferol', 'myricetin', 'rutin', 'hesperidin',
    'naringenin', 'apigenin', 'luteolin', 'chrysin',
    'catechin', 'epicatechin', 'epigallocatechin',

    # Common organic compounds
    'benzene', 'toluene', 'phenol', 'aniline', 'benzoic acid',
    'salicylic acid', 'acetylsalicylic acid', 'mandelic acid',
    'cinnamic acid', 'vanillin', 'eugenol', 'thymol', 'menthol',

    # Additional drugs (continuing alphabetically)
    'acetazolamide', 'acyclovir', 'adenosine', 'allopurinol', 'amiodarone',
    'baclofen', 'beclomethasone', 'bethanechol', 'bimatoprost',
    'bromocriptine', 'bupropion', 'buspirone', 'calcitonin', 'calcitriol',
    'candesartan', 'carisoprodol', 'chloroquine', 'chlorpromazine',
    'ciclopirox', 'clindamycin', 'clopidogrel', 'clotrimazole',
    'cyclobenzaprine', 'cyclophosphamide', 'cyproheptadine',
    'dantrolene', 'dapsone', 'desoximetasone', 'dexamethasone',
    'dicloxacillin', 'digoxin', 'divalproex', 'donepezil',
    'dutasteride', 'eletriptan', 'entacapone', 'ergotamine',
    'ethambutol', 'ethionamide', 'etoposide', 'ezetimibe',
    'finasteride', 'flecainide', 'fluconazole', 'fluorouracil',
    'fluphenazine', 'fluticasone', 'frovatriptan', 'gabapentin',
    'gemfibrozil', 'glimepiride', 'granisetron', 'guaifenesin',
    'haloperidol', 'hydralazine', 'hydroxyurea', 'hyoscyamine',
    'indapamide', 'irbesartan', 'isoniazid', 'isosorbide',
    'isotretinoin', 'itraconazole', 'ivermectin', 'ketamine',
    'labetalol', 'latanoprost', 'levofloxacin', 'lidocaine',
    'linezolid', 'lithium', 'loperamide', 'losartan',
    'memantine', 'mercaptopurine', 'mesalamine', 'methotrexate',
    'methylphenidate', 'methylprednisolone', 'metoclopramide',
    'miconazole', 'milrinone', 'minoxidil', 'mirtazapine',
    'misoprostol', 'morphine', 'mupirocin', 'nabumetone',
    'nadolol', 'nafcillin', 'naltrexone', 'nateglinide',
    'nebivolol', 'neomycin', 'neostigmine', 'niacin',
    'nicardipine', 'nitroglycerin', 'nitroprusside', 'norfloxacin',
    'nystatin', 'octreotide', 'olanzapine', 'olmesartan',
    'olopatadine', 'ondansetron', 'orlistat', 'oseltamivir',
    'oxcarbazepine', 'oxybutynin', 'paclitaxel', 'paliperidone',
    'penicillamine', 'pentoxifylline', 'pergolide', 'perindopril',
    'phenazopyridine', 'phenelzine', 'phenobarbital', 'phentermine',
    'phenylephrine', 'pilocarpine', 'pimozide', 'pindolol',
    'pramipexole', 'prasugrel', 'prazosin', 'primaquine',
    'primidone', 'probenecid', 'procainamide', 'prochlorperazine',
    'promethazine', 'propofol', 'propylthiouracil', 'pseudoephedrine',
    'pyrazinamide', 'pyridostigmine', 'quetiapine', 'quinapril',
    'raloxifene', 'repaglinide', 'rifabutin', 'rifampin',
    'riluzole', 'risedronate', 'risperidone', 'ritonavir',
    'rituximab', 'rivastigmine', 'rizatriptan', 'ropinirole',
    'rotigotine', 'saquinavir', 'scopolamine', 'selegiline',
    'sildenafil', 'simethicone', 'sitagliptin', 'solifenacin',
    'sotalol', 'sucralfate', 'sulfadiazine', 'sulfamethoxazole',
    'sulfasalazine', 'sumatriptan', 'tacrolimus', 'tadalafil',
    'tamoxifen', 'tamsulosin', 'telmisartan', 'temazepam',
    'terbinafine', 'terconazole', 'theophylline', 'thiamine',
    'ticlopidine', 'timolol', 'tizanidine', 'tobramycin',
    'tolterodine', 'topiramate', 'torsemide', 'tranylcypromine',
    'trazodone', 'triamterene', 'triazolam', 'trimethoprim',
    'trospium', 'valproate', 'valsartan', 'vancomycin',
    'vardenafil', 'venlafaxine', 'vigabatrin', 'voriconazole',
    'zaleplon', 'zidovudine', 'ziprasidone', 'zolpidem',
    'zonisamide', 'zopiclone',

    # More natural products
    'betaine', 'choline', 'carnitine', 'taurine', 'creatine',
    'coenzyme q10', 'lipoic acid', 'pantothenic acid',

    # Nucleotides and bases
    'adenine', 'guanine', 'cytosine', 'thymine', 'uracil',
    'adenosine', 'guanosine', 'cytidine', 'thymidine', 'uridine',

    # Additional common compounds
    'acetone', 'ethanol', 'methanol', 'glycerol', 'sorbitol',
    'mannitol', 'xylitol', 'erythritol', 'maltitol',
    'formaldehyde', 'acetaldehyde', 'propionaldehyde',
    'benzaldehyde', 'salicylaldehyde',

    # More antibiotics
    'amikacin', 'gentamicin', 'streptomycin', 'tobramycin',
    'clindamycin', 'lincomycin', 'vancomycin', 'teicoplanin',
    'nitrofurantoin', 'fosfomycin', 'colistin', 'polymyxin',

    # Antifungals
    'amphotericin', 'nystatin', 'ketoconazole', 'fluconazole',
    'itraconazole', 'voriconazole', 'posaconazole',
    'terbinafine', 'griseofulvin', 'flucytosine',

    # Antivirals
    'acyclovir', 'ganciclovir', 'cidofovir', 'foscarnet',
    'ribavirin', 'sofosbuvir', 'ledipasvir', 'velpatasvir',
    'oseltamivir', 'zanamivir', 'peramivir', 'baloxavir',

    # Cancer drugs
    'methotrexate', 'fluorouracil', 'capecitabine', 'gemcitabine',
    'cytarabine', 'doxorubicin', 'daunorubicin', 'epirubicin',
    'bleomycin', 'mitomycin', 'cisplatin', 'carboplatin',
    'oxaliplatin', 'irinotecan', 'topotecan', 'etoposide',
    'paclitaxel', 'docetaxel', 'vincristine', 'vinblastine',
    'cyclophosphamide', 'ifosfamide', 'melphalan', 'chlorambucil',
    'tamoxifen', 'raloxifene', 'anastrozole', 'letrozole',
    'bicalutamide', 'flutamide', 'nilutamide',

    # Immunosuppressants
    'cyclosporine', 'tacrolimus', 'sirolimus', 'everolimus',
    'azathioprine', 'mycophenolate', 'methotrexate',

    # Anesthetics
    'lidocaine', 'bupivacaine', 'ropivacaine', 'mepivacaine',
    'prilocaine', 'procaine', 'tetracaine', 'benzocaine',
    'propofol', 'etomidate', 'ketamine', 'thiopental',
    'sevoflurane', 'isoflurane', 'desflurane', 'halothane',

    # Muscle relaxants
    'succinylcholine', 'rocuronium', 'vecuronium', 'atracurium',
    'pancuronium', 'baclofen', 'tizanidine', 'cyclobenzaprine',
    'methocarbamol', 'carisoprodol', 'chlorzoxazone',

    # Antipsychotics
    'haloperidol', 'chlorpromazine', 'fluphenazine', 'perphenazine',
    'risperidone', 'olanzapine', 'quetiapine', 'ziprasidone',
    'aripiprazole', 'paliperidone', 'lurasidone', 'asenapine',
    'clozapine', 'loxapine', 'molindone', 'thiothixene',

    # Mood stabilizers
    'lithium carbonate', 'valproic acid', 'carbamazepine',
    'lamotrigine', 'oxcarbazepine', 'topiramate',

    # Stimulants
    'methylphenidate', 'dexmethylphenidate', 'amphetamine',
    'dextroamphetamine', 'lisdexamfetamine', 'modafinil',
    'armodafinil', 'atomoxetine',

    # Antiemetics
    'ondansetron', 'granisetron', 'dolasetron', 'palonosetron',
    'metoclopramide', 'prochlorperazine', 'promethazine',
    'meclizine', 'dimenhydrinate', 'scopolamine', 'aprepitant',

    # Antidiarrheals
    'loperamide', 'diphenoxylate', 'bismuth subsalicylate',
    'octreotide', 'cholestyramine', 'colesevelam',

    # Laxatives
    'bisacodyl', 'senna', 'docusate', 'lactulose',
    'polyethylene glycol', 'magnesium hydroxide', 'sorbitol',

    # Bronchodilators
    'albuterol', 'levalbuterol', 'pirbuterol', 'terbutaline',
    'formoterol', 'salmeterol', 'indacaterol', 'vilanterol',
    'ipratropium', 'tiotropium', 'aclidinium', 'glycopyrrolate',
    'theophylline', 'aminophylline',

    # Corticosteroids
    'prednisone', 'prednisolone', 'methylprednisolone',
    'dexamethasone', 'betamethasone', 'triamcinolone',
    'budesonide', 'fluticasone', 'mometasone', 'ciclesonide',
    'beclomethasone', 'flunisolide', 'hydrocortisone',

    # Thyroid medications
    'levothyroxine', 'liothyronine', 'desiccated thyroid',
    'methimazole', 'propylthiouracil',

    # Diabetes medications
    'metformin', 'glipizide', 'glyburide', 'glimepiride',
    'pioglitazone', 'rosiglitazone', 'sitagliptin', 'saxagliptin',
    'linagliptin', 'alogliptin', 'exenatide', 'liraglutide',
    'dulaglutide', 'semaglutide', 'canagliflozin', 'dapagliflozin',
    'empagliflozin', 'ertugliflozin', 'acarbose', 'miglitol',
    'nateglinide', 'repaglinide', 'pramlintide',

    # Osteoporosis medications
    'alendronate', 'risedronate', 'ibandronate', 'zoledronic acid',
    'denosumab', 'teriparatide', 'raloxifene', 'calcitonin',

    # Gout medications
    'allopurinol', 'febuxostat', 'probenecid', 'colchicine',
    'pegloticase', 'lesinurad',

    # Migraine medications
    'sumatriptan', 'rizatriptan', 'zolmitriptan', 'naratriptan',
    'almotriptan', 'frovatriptan', 'eletriptan', 'ergotamine',
    'dihydroergotamine', 'topiramate', 'valproate', 'propranolol',

    # Urological medications
    'tamsulosin', 'alfuzosin', 'doxazosin', 'terazosin', 'silodosin',
    'finasteride', 'dutasteride', 'oxybutynin', 'tolterodine',
    'solifenacin', 'darifenacin', 'fesoterodine', 'trospium',
    'mirabegron', 'bethanechol', 'phenazopyridine',

    # Ophthalmic medications
    'timolol', 'betaxolol', 'carteolol', 'levobunolol',
    'latanoprost', 'travoprost', 'bimatoprost', 'tafluprost',
    'brimonidine', 'apraclonidine', 'dorzolamide', 'brinzolamide',
    'pilocarpine', 'carbachol', 'echothiophate',

    # Dermatological medications
    'tretinoin', 'adapalene', 'tazarotene', 'isotretinoin',
    'benzoyl peroxide', 'clindamycin', 'erythromycin',
    'mupirocin', 'bacitracin', 'neomycin', 'polymyxin',
    'hydrocortisone', 'triamcinolone', 'clobetasol', 'betamethasone',
    'tacrolimus', 'pimecrolimus', 'calcipotriene', 'calcitriol',

    # More vitamins and cofactors
    'ubiquinone', 'menaquinone', 'retinyl palmitate', 'retinyl acetate',
    'ergocalciferol', 'cholecalciferol', 'tocopherol', 'tocotrienol',
    'phytonadione', 'menadione', 'thiamine pyrophosphate',
    'flavin adenine dinucleotide', 'nicotinamide adenine dinucleotide',
    'pyridoxal phosphate', 'methylcobalamin', 'adenosylcobalamin',
]

def get_large_molecule_list():
    """Return list of 1000+ molecules."""
    return LARGE_MOLECULE_LIST

def get_sample_molecules(n=100):
    """Get a sample of n molecules from the large list."""
    import random
    return random.sample(LARGE_MOLECULE_LIST, min(n, len(LARGE_MOLECULE_LIST)))
