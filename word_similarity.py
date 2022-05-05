#resume modeling:
import pandas as pd




keywords = {'ux,designer': ['design',
  'product',
  'user',
  'ui',
  'ux',
  'research',
  'custom',
  'wireframing',
  'user-centered Design',
  'experience design',
  'information architecture'
  'platform'],
 'data,scientist': ['data',
 'algorithm',
 'analytics'
  'model',
  'statistics',
  'machine learning',
  'recommendation',
  'Hadoop',
  'analysis','python'],
 'data,analyst': ['data',
  'tableau',
  'sql',
  'analysis',
  'data visualization',
  'report',
  'analyst',
  'algorithm',
  'microsoft access',
  'report','python'],
 'project,manager': ['project',
  'manager',
  'agile',
  'develop',
  'budgeting',
  'growth',
  'assist',
  'business alignment'],
 'product,manager': ['product',
  'product launch',
  'manager',
  'product management',
  'business',
  'scrum',
  'marketing strategy',
  'solution',
  'build'],
 'account,manager': ['client',
  'manager',
  'account',
  'business-to-business(B2B)',
  'sales management',
  'marketing strategy',
  'sale',
  'device',
  'strategies',
  'digital',
  'business strategy'],
 'consultant': ['consultant',
  'client',
  'market',
  'obtained backing or support',
  'business',
  'manager',
  'project',
  'association',
  'process',
  'solution',
  'architecture'],
 'marketing': ['market',
 'Brand management','Market research',
'Strategic marketing plans',
'Event marketing',
'Social media marketing',
'Distribution channels',
'Product launch',
'Public relations',
  'campaign',
  'content',
  'brand',
  'growth'],
 'sales': ['sale',
 'accounts',
'business',
'clients',
'CMS',
'communication',
'CRM',
  'inventory']}
  

import process_data as pda



def resume_reader(example, keyword):
    toke_example = pda.tokenize_stem(example)
    example_words = toke_example[0].split()
    matching_words = []
    missing_words = []

    for word in keywords[keyword]:
        if word in example_words:
         
           
          matching_words.append(word)
        else:
            missing_words.append(word)
    return set(matching_words), set(missing_words)
