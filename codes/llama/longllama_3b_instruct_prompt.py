import numpy as np
import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM, TextStreamer, PreTrainedModel, PreTrainedTokenizer
from typing import List, Optional

MODEL_PATH = (
    "syzymon/long_llama_3b_instruct"
)
TOKENIZER_PATH = MODEL_PATH
# to fit into colab GPU we will use reduced precision
TORCH_DTYPE = torch.bfloat16

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_DTYPE,
    device_map=device,
    trust_remote_code=True,
    # mem_attention_grouping is used
    # to trade speed for memory usage
    # for details, see the section Additional configuration
    mem_attention_grouping=(1, 2048),
)
model.eval()

@torch.no_grad()
def load_to_memory(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, text: str):
    tokenized_data = tokenizer(text, return_tensors="pt")
    input_ids = tokenized_data.input_ids
    input_ids = input_ids.to(model.device)
    torch.manual_seed(0)
    output = model(input_ids=input_ids)
    memory = output.past_key_values
    return memory


@torch.no_grad()
def generate_with_memory(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, memory, prompt: str, temperature=0.3
):
    tokenized_data = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized_data.input_ids
    input_ids = input_ids.to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=False)

    new_memory = memory

    stop = False
    while not stop:
        output = model(input_ids, past_key_values=new_memory)
        new_memory = output.past_key_values
        assert len(output.logits.shape) == 3
        assert output.logits.shape[0] == 1
        last_logit = output.logits[[0], [-1], :]
        dist = torch.distributions.Categorical(logits=last_logit / temperature)
        next_token = dist.sample()
        if next_token[0] == tokenizer.eos_token_id:
            streamer.put(next_token[None, :])
            streamer.end()
            stop = True
        else:
            input_ids = next_token[None, :]
            streamer.put(input_ids)


PROMPT_PREFIX = "You are an AI assistant. User will you give you a task. Your goal is to complete the task as faithfully as you can.\n"


def construct_question_prompt(question: str):
    prompt = f"\nAnswer the following questions using information from the text above.\nQuestion: {question}\nAnswer: "
    return prompt


def ask_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt: str, memory, seed=0):
    tokenized_data = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized_data.input_ids
    input_ids = input_ids.to(model.device)

    torch.manual_seed(seed)
    generate_with_memory(model, tokenizer, memory, prompt)

article = """
Q1: Generalized Anxiety Disorder

Essential (Required) Features:
 
Marked symptoms of anxiety accompanied by either:
general apprehensiveness that is not restricted to any particular environmental circumstance (i.e., “free-floating anxiety”); or
worry (apprehensive expectation) about untoward events occurring in several different aspects of everyday life (e.g., work, finances, health, family).
 
Anxiety and general apprehensiveness or worry are accompanied by additional symptoms, such as:
Muscle tension or motor restlessness. Sympathetic autonomic overactivity as evidenced by frequent gastrointestinal symptoms such as nausea and/or abdominal distress, heart palpitations, sweating, trembling, shaking, and/or dry mouth. Subjective experience of nervousness, restlessness, or being “on edge”. Difficulties maintaining concentration. Irritability. Sleep disturbances (difficulty falling or staying asleep, or restless, unsatisfying sleep).
 
The symptoms are not transient and persist for at least several months, for more days than not. The symptoms are sufficiently severe to result in significant distress about experiencing persistent anxiety symptoms or result in significant impairment in personal, family, social, educational, occupational, or other important areas of functioning.

Boundaries with Other Disorders and Normality:
 
Anxiety and worry are normal emotional/cognitive states that commonly occur when people are under stress. At optimal levels, anxiety and worry may help to direct problem-solving efforts, focus attention adaptively, and increase alertness. Normal anxiety and worry are usually sufficiently self-regulated that they do not interfere with functioning or cause marked distress. In contrast, the anxiety and worry characteristic of Generalized Anxiety Disorder are excessive, persistent, intense, and may have a significant negative impact on functioning. Individuals under extremely stressful circumstances (e.g., living in a war zone) may experience intense and impairing anxiety and worry that is appropriate to their environmental circumstances, and these experiences should not be regarded as symptomatic of Generalized Anxiety Disorder if they occur only under such circumstances.  (Boundary with normality). The symptoms are not a manifestation of a medical disorder that is not classified under Mental and Behavioural Disorders (e.g., hyperthyroidism) and are not due to the effect of a substance or medication on the central nervous system (e.g., coffee, cocaine), including withdrawal effects (e.g., alcohol, benzodiazepines). (Boundary with Substance-Induced Anxiety Disorder and Anxiety Disorder Due to a Disorder or Disease Classified Elsewhere). Generalized Anxiety Disorder and Depressive Disorders can share several features such as somatic symptoms of anxiety, difficulty with concentration, sleep disruption, and feelings of dread associated with pessimistic thoughts. Depressive Disorders are differentiated by the presence of low mood or loss of pleasure in previously enjoyable activities and other characteristic symptoms of Depressive Disorders (e.g., appetite changes, feelings of worthlessness, suicidal ideation). Generalized Anxiety Disorder may co-occur with Depressive Disorders, but should only be diagnosed if the definitional requirements of Generalized Anxiety Disorder were met prior to the onset of or following complete remission of a Depressive Episode. (Boundary with Depressive Disorders)
 
Adjustment Disorder involves maladaptive reactions to an identifiable psychosocial stressor or multiple stressors characterized by preoccupation with the stressor or its consequences. Reactions may include excessive worry, recurrent and distressing thoughts about the stressor, or constant rumination about its implications. Adjustment Disorder centers on the identifiable stressor or its consequences, whereas in GAD, worry typically encompasses multiple areas of daily life and may include hypothetical concerns (e.g., that a negative life event may occur). Unlike individuals with GAD, those with Adjustment disorder typically have normal functioning prior to the onset of the stressor(s). Symptoms of Adjustment Disorder generally resolve within 6 months. (Boundary with Adjustment Disorder). In Social Anxiety Disorder, symptoms occur in response to feared social situations (e.g., speaking in public, initiating a conversation) and the primary focus of apprehension is being negatively evaluated by others. Individuals with Generalized Anxiety Disorder may worry about the implications of performing poorly or failing an examination, but are not exclusively concerned about being evaluated by others. (Boundary with Social Anxiety Disorder)
 
Panic Disorder is characterized by recurrent, unexpected, self-limited episodes of intense fear or anxiety. Generalized Anxiety Disorder is differentiated by a more persistent and less circumscribed chronic feeling of apprehensiveness usually associated with worry about a variety of different everyday life events. Individuals with Generalized Anxiety Disorder may experience panic attacks that are triggered by specific worries. In such cases, an additional diagnosis of Panic Disorder should only be given if the person has also experienced panic attacks without specific provocation (i.e., unexpected panic attacks). Moreover, some individuals with Panic Disorder may experience anxiety and worry between panic attacks. If the focus of the anxiety and worry is confined to fear of having a panic attack or the possible implications of panic attacks (e.g., that the individual may be suffering from a cardiovascular illness), an additional diagnosis of Generalized Anxiety Disorder is not warranted. If, however, the individual is more generally anxious about a number of life events in addition to experiencing unprovoked panic attacks, an additional diagnosis of Generalized Anxiety Disorder may be appropriate. (Boundary with Panic Disorder)
 
Individuals with Posttraumatic Stress Disorder develop hypervigilance as a consequence of exposure to the traumatic stressor and may become apprehensive that they or others close to them may be under immediate threat either in specific situations or more generally. Individuals with Posttraumatic Stress Disorder may also experience anxiety triggered by reminders of the traumatic event (e.g., fear and avoidance of a place where an individual was assaulted). In contrast, the anxiety and worry in individuals with Generalized Anxiety Disorder is directed toward the possibility of untoward events in a variety of life domains (e.g., health, finances, work). (Boundary with Posttraumatic Stress Disorder)
 
In Obsessive-Compulsive Disorder, the focus of apprehension is on intrusive and unwanted thoughts, urges, or images (obsessions), whereas in Generalized Anxiety Disorder the focus is on everyday life events. In contrast to obsessions in Obsessive-Compulsive Disorder, which are usually experienced as unwanted and intrusive, individuals with Generalized Anxiety Disorder may experience their worry as a strategy that is helpful in averting negative outcomes. (Boundary with Obsessive-Compulsive Disorder). Individuals with Generalized Anxiety Disorder may worry about the health and safety of attachment figures, as in Separation Anxiety Disorder, but their worry also extends to other aspects of everyday life events.  (Boundary with Separation Anxiety Disorder)

Additional Features: Some individuals with Generalized Anxiety Disorder may only report chronic somatic anxiety and a feeling of nonspecific dread without being able to articulate specific worry content. Behavioural changes such as avoidance, frequent need for reassurance (especially in children), and procrastination may be seen. These behaviours typically represent an effort to reduce apprehension or prevent untoward events from occurring.

Q2: Panic Disorder

Essential (Required) Features:
 
Recurrent unexpected panic attacks that are not restricted to particular stimuli or situations. Panic attacks are discrete episodes of intense fear or apprehension also characterized by the rapid and concurrent onset of several characteristic symptoms. These symptoms may include, but are not limited to, the following:
Palpitations or increased heart rate. Sweating. Trembling. Sensations of shortness of breath. Feelings of choking. Chest pain. Nausea or abdominal distress. Feelings of dizziness or lightheadedness. Chills or hot flushes. Tingling or lack of sensation in extremities (i.e., paresthesias). Depersonalization or derealization. Fear of losing control or going mad. Fear of imminent death
 
Panic attacks are followed by persistent concern or worry (e.g., for several weeks) about their recurrence or their perceived negative significance (e.g., that the physiological symptoms may be those of a myocardial infarction), or behaviours intended to avoid their recurrence (e.g., only leaving the home with a trusted companion). The symptoms are sufficiently severe to result in significant impairment in personal, family, social, educational, occupational, or other important areas of functioning. Panic attacks can occur in other Anxiety and Fear-Related Disorders as well as other Mental and Behavioural Disorders and therefore the presence of panic attacks is not in itself sufficient to assign a diagnosis of Panic Disorder.
 
Boundaries with Other Disorders and Normality:
 
The sudden onset, rapid peaking, unexpected nature, and intense severity of panic attacks differentiates them from normal situation-bound anxiety that may be experienced in everyday life (e.g., during stressful life transitions such as moving to a new city). Furthermore, Panic Disorder is differentiated from normal fear reactions by frequent recurrence of panic attacks, persistent alterations in behaviour following panic attacks (e.g., avoidance), and associated significant impairment in functioning. Occasionally, an individual may experience a single, isolated panic attack and never experience a recurrence. A diagnosis is not warranted in such cases. (Boundary with normality). The symptoms are not a manifestation of a medical disorder that is not classified under Mental and Behavioural Disorders (e.g., pheochromocytoma) and are not due to the direct effects of a substance or medication on the central nervous system (e.g., coffee, cocaine), including withdrawal effects (e.g., alcohol, benzodiazepines). (Boundary with Substance-Induced Anxiety Disorder and Anxiety Disorder Due to a Disorder or Disease Classified Elsewhere)
 
Panic attacks can occur in the context of a variety of other Mental and Behavioural Disorders, particularly other Anxiety and Fear-Related Disorders, Disorders Specifically Associated with Stress, and Obsessive-Compulsive and Related Disorders. When panic attacks occur in the context of these disorders, they are generally part of an intense anxiety response to a distressing internal or external stimulus that represents a focus of apprehension in that disorder (e.g., a particular object or situation in Specific Phobia, negative social evaluation in Social Anxiety Disorder, being contaminated by germs in Obsessive-Compulsive Disorder, having a serious illness in Hypochondriasis, reminders of a traumatic event in Posttraumatic Stress Disorder). If panic attacks are limited to such situations in the context of another disorder, a separate diagnosis of Panic Disorder is not warranted. If some panic attacks over the course of the disorder have been unexpected and not exclusively in response to stimuli associated with the focus of apprehension related to another disorder, an additional diagnosis of Panic Disorder may be assigned. (Boundary with Anxiety Disorders and other Mental and Behavioural Disorders). The perceived unpredictability of panic attacks often reflects the early phase of the illness. However, over time, with the reoccurrence of panic attacks in specific situations, individuals often develop anticipatory anxiety about having panic attacks in those situations or may experience panic attacks triggered by exposure to them. If the individual develops fears that panic-like or other incapacitating or embarrassing symptoms will occur in multiple situations, and as a result actively avoids these situations, requires the presence of a companion, or endures them only with intense fear or anxiety and all other definitional requirements of Agoraphobia are met, he or she may qualify for an additional diagnosis of Agoraphobia. (Boundary with Agoraphobia). Individuals with Hypochondriasis often misinterpret bodily symptoms as evidence that they may have one or more life threatening illnesses. Although individuals with Panic Disorder may also manifest concerns that physical manifestations of anxiety are indicative of life threatening illnesses (e.g., myocardial infarction), these symptoms typically occur in the midst of a panic attack. Individuals with Panic Disorder are more concerned about the recurrence of panic attacks or panic-like symptoms, are less likely to report somatic concerns attributable to bodily symptoms other than those associated with anxiety, and are less likely to engage in repetitive and excessive health-related behaviours. However, panic attacks can occur in Hypochondriasis and if they are exclusively associated with fears of having a life-threatening illness, an additional diagnosis of Panic Disorder it not warranted. However, the two disorders can co-occur, and if there are persistent and repetitive panic attacks that are not in response to illness-related concerns, both diagnoses should be assigned. (Boundary with Hypochondriasis). Panic attacks can occur in Depressive Disorders, particularly in those with Prominent Anxiety Symptoms as well as in Mixed Depressive and Anxiety Disorder, and may be triggered by depressive ruminations. If unexpected panic attacks occur in the context of these disorders and the main concern is about the perceived dangerousness of the panic-like symptoms, an additional diagnosis of Panic Disorder may be appropriate. (Boundary with Depressive Disorders and Mixed Depressive and Anxiety Disorder)

Additional Features:
 
Individual panic attacks usually only last for minutes, though some may last longer. The frequency and severity of panic attacks varies widely (e.g., many times a day to a few per month) within and across individuals. Limited-symptom attacks (i.e., attacks that are similar to panic attacks, except that they are accompanied by only a few symptoms characteristic of a panic attack without the characteristic intense peak of symptoms) are common in individuals with Panic Disorder, particularly as behavioural strategies (e.g., avoidance) are used to curtail anxiety symptoms. However, in order to qualify for a diagnosis of Panic Disorder, there must be a history of recurrent panic attacks that meet the full definitional requirements. A diagnosis of Agoraphobia without the additional diagnosis of Panic Disorder can be considered if there is no history of recurrent full symptom panic attacks and all definitional requirements are met for Agoraphobia. Some individuals with Panic Disorder experience nocturnal panic attacks, that is, waking from sleep in a state of panic.
 
Although the pattern of symptoms (e.g., mainly respiratory, nocturnal, etc.), the severity of the anxiety, and the extent of avoidance behaviours are variable, Panic Disorder is one of the most impairing of the Anxiety Disorders. Individuals often present repeatedly to emergency units and may undergo a range of unnecessary and costly special medical investigations despite repeated negative findings.

Q3: Agoraphobia

Essential (Required) Features:
 
Marked and excessive fear or anxiety that occurs in, or in anticipation of, multiple situations where escape might be difficult or help might not be available, such as using public transportation, being in crowds, being outside the home alone, in shops, theatres, or standing in line. The individual is consistently fearful or anxious about these situations due to a sense of danger or fear of specific negative outcomes such as panic attacks, symptoms of panic, or other incapacitating (e.g., falling) or embarrassing physical symptoms (e.g., incontinence). The situations are actively avoided, are entered only under specific circumstances (e.g., in the presence of a companion), or else are endured with intense fear or anxiety. The symptoms are not transient, that is, they persist for an extended period of time (e.g., at least several months). The symptoms are sufficiently severe to result in significant distress about experiencing persistent anxiety symptoms or significant impairment in personal, family, social, educational, occupational, or other important areas of functioning.
 
Boundaries with Other Disorders and Normality:
 
Individuals may exhibit transient avoidance behaviours in the context of normal development or in periods of increased stress. These behaviours are differentiated from Agoraphobia because they are limited in duration and do not lead to significant impact on functioning. (Boundary with normality). Individuals with other medical disorders not classified under Mental and Behavioural Disorders may demonstrate avoidance because of reasonable concerns about being incapacitated (e.g., mobility limitations in an individual with a neurological disorder) or embarrassed (e.g., diarrhea in an individual with Crohn’s disease). Agoraphobia should only be diagnosed if the fear or anxiety and avoidance result in greater functional impairment as compared to others who have a similar health condition. (Boundary with normality). Specific Phobia is differentiated from Agoraphobia because it involves fear of circumscribed situations or stimuli themselves (e.g., heights, animals, blood or injury) rather than fear or anxiety of imminent perceived dangerous outcomes (e.g., panic attacks, symptoms of panic, incapacitation, or embarrassing physical symptoms) that are anticipated to occur in multiple situations where obtaining help or escaping might be difficult. (Boundary with Specific Phobia). In Social Anxiety Disorder, symptoms in response to feared social situations (e.g., speaking in public, initiating a conversation) and the primary focus of apprehension is on being negatively evaluated by others. (Boundary with Social Anxiety Disorder). In Posttraumatic Stress Disorder, the individual deliberately avoids reminders likely to produce re-experiencing of the traumatic event(s). In contrast, situations are avoided in Agoraphobia because of fear or anxiety of imminent perceived dangerous outcomes (e.g., panic attacks, symptoms of panic, incapacitation, or embarrassing physical symptoms) that are anticipated to occur in multiple situations where obtaining help or escaping might be difficult. (Boundary with Posttraumatic Stress Disorder). Similar to Agoraphobia, individuals with Separation Anxiety Disorder avoid situations but, in contrast, they do so to prevent or limit being away from individuals to whom they are attached (e.g., parent, spouse, or child) for fear of losing them. (Boundary with Separation Anxiety Disorder). Individuals with Generalized Anxiety Disorder may avoid situations but do so to avert feared negative consequences in everyday life situations (e.g., avoiding going on a family trip because of worries that a family member will get sick) rather than because of fear or anxiety of imminent perceived dangerous outcomes (e.g., panic attacks, symptoms of panic, incapacitation, or embarrassing physical symptoms) that are anticipated to occur in multiple situations where obtaining help or escaping might be difficult. (Boundary with Generalized Anxiety Disorder). Many individuals with Agoraphobia have also experienced recurrent panic attacks. If the panic attacks occur exclusively in the context of the multiple agoraphobic situations without the presence of unexpected panic attacks, an additional diagnosis of Panic Disorder is not warranted. However, if unexpected panic attacks also occur and all other diagnostic requirements of Panic Disorder are met, both diagnoses may be assigned. (Boundary with Panic Disorder). In Depressive Disorders, individuals may avoid mulitple situations but do so because of loss of interest in previously pleasurable activities or due to lack of energy rather than because of fear or anxiety of imminent perceived dangerous outcomes (e.g., panic attacks, symptoms of panic, incapacitation, or embarrassing physical symptoms) that are anticipated to occur in multiple situations where obtaining help or escaping might be difficult. (Boundary with Depressive Disorders)

Additional Features:
 
The experiences feared by individuals with Agoraphobia include any of the symptoms of a panic attack as decribed in Panic Disorder (e.g., palpitations or increased heart rate, chest pain, feelings of dizziness or lightheadedness) or other symptoms that may be incapacitating, frightening, difficult to manage, or embarrassing (e.g., incontinence, changes in vision, vomiting).
 
Individuals with Agoraphobia may employ a variety of different behavioural strategies if required to enter feared situations. One such ‘safety’ behaviour is to require the presence of a companion. Other strategies may include going to certain places only at particular times of day or carrying specific materials (e.g., medications, towels) in case of the feared negative outcome. These strategies may change over the course of the disorder and from one occasion to the next. For example, on different occasions in the same situation an individual may insist on having a companion, endure the situation with distress, or use various safety behaviours to cope with his or her anxiety.
 
Although the pattern of symptoms, the severity of the anxiety, and the extent of avoidance are variable, Agoraphobia is one of the most impairing of the Anxiety Disorders to the extent that some individuals become completely housebound, which has an impact on opportunities for employment, seeking medical care, and the ability to form and maintain relationships.

Vignette:
MT is a 23-year-old woman who lives alone and is a graduate student at a local university. She was referred for psychiatric evaluation by an emergency room physician, at the university hospital, who attended her during her second visit to the emergency department within a 2-month period. According to hospital records, on both occasions MT complained of severe chest pain, dizziness, and a tingling sensation in her arms and legs and thought she was having a heart attack. 

Presenting Symptoms

During the psychiatric evaluation, MT explains that she has experienced several big life changes recently, including breaking up with her boyfriend and moving to her current city 4 months ago to enroll in graduate school. However, she reports that the episodes of being “extremely fearful and feeling like she was dying” started approximately 9 months before that, prior to the breakup and her move. The first time, she was out walking with her father and suddenly started feeling extremely apprehensive and fearful, “for no reason.”  Over the next few minutes, as the feeling got worse, she began to feel dizzy and started having trouble breathing. She says her heart was racing and her fingertips started tingling, and she felt a tightness in her chest. She thought she was having a heart attack and was going to die, so her father rushed her to the nearest emergency room. After 30 minutes in the hospital, she “felt okay again,” but was shaken by what had happened. After several examinations, she was found to be in good health and doctors could not identify any cardiovascular abnormality, or other cause, for her symptoms. 

Since then, MT has had multiple similar episodes of sudden, intense feelings of anxiety, “like the world is about to end,” and the same physiological sensations.  She is not able to identify any precipitating event for the episodes; they have usually happened when she has been on a walk or studying. She indicates that she was not thinking about any particular topic when the episodes occurred. She finds herself worrying about having another episode to such an extent that she has major difficulty focusing on her schoolwork.

She says that her move has made her even more terrified that she would have another episode, and that the effects would be even worse because she does not have family or close friends nearby. So, when she had her first episode after moving, she went to the university hospital. “I thought I was dying and I didn’t know what else to do.” During her most recent trip to the university hospital emergency room, the attending physician recognized her and told her that there was nothing wrong with her heart, and that she needed to see a psychiatrist. She states that she decided to attend this psychiatric evaluation because, not only is she exhausted by the emotional impact of her condition, but the mounting costs of her ER visits have become so burdensome that she is unable to pay her university fees.

Additional Background Information

MT reports that she drinks alcohol occasionally, consuming two or three drinks with friends on weekends, but says that she does not smoke or use substances. She is doing well in school, but says that she is afraid that she will have to leave school and move back home if the episodes continue. “That is, if they don’t kill me.”
"""

fot_memory = load_to_memory(model, tokenizer, PROMPT_PREFIX + article)

initial_prompt = """
Based on the vignette provided, reference the guidelines provided (Q1, Q2, Q3) and choose among the following disorders: Generalized Anxiety Disorder, Agoraphobia, and Panic Disorder. Explain your choice by referring to the symptoms of the disorder and how they relate to the vignette. Also explain why they do not relate to the other disorders. You must explain why they do not relate to other disorders.
"""
prompt = construct_question_prompt(initial_prompt)

ask_model(model, tokenizer, prompt, fot_memory)