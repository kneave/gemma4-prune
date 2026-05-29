#!/home/kitt/ai/gemma4-prune-venv/bin/python
"""
Quick test: do different languages activate different MoE experts?

Runs 10 samples each in 10 languages through the model, comparing
which experts each layer's router selects. Uses the full multimodal
model (not just text_model) to get proper chat template handling.

Usage:
    python lang_test.py --model-path /home/kitt/gemma4-prune/bf16
"""

import argparse
import json
import os
import gc
import time
from pathlib import Path

from collections import defaultdict
import torch
import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
torch.set_num_threads(4)

LANGUAGES = {
    "english": [
        "The development of renewable energy sources has become increasingly important as concerns about climate change grow.",
        "In a small village nestled between rolling hills, the residents gathered for their annual harvest festival.",
        "Modern architecture often emphasizes open floor plans and sustainable materials that reduce environmental impact.",
        "The orchestra performed a stunning rendition of Beethoven's Ninth Symphony that brought the audience to their feet.",
        "Scientific research requires rigorous methodology and peer review to ensure the validity of published findings.",
        "The hiking trail wound through dense forest before opening up to a breathtaking view of the mountain valley.",
        "Digital privacy remains a contentious issue as governments balance security concerns with individual rights.",
        "Economic indicators suggest that the recovery from the recession is proceeding faster than analysts initially predicted.",
        "The chef carefully plated the dish, adding microgreens and a drizzle of truffle oil for the final touch.",
        "Educational reformers argue that critical thinking skills should be prioritized over rote memorization in schools.",
    ],
    "french": [
        "Le développement des sources d'énergie renouvelables est devenu de plus en plus important face aux préoccupations croissantes concernant le changement climatique.",
        "Dans un petit village niché entre des collines ondulantes, les habitants se sont réunis pour leur festival annuel de la récolte.",
        "L'architecture moderne met souvent l'accent sur les plans d'étage ouverts et les matériaux durables qui réduisent l'impact environnemental.",
        "L'orchestre a interprété une version saisissante de la Neuvième Symphonie de Beethoven qui a levé le public de ses sièges.",
        "La recherche scientifique exige une méthodologie rigoureuse et une évaluation par les pairs pour garantir la validité des résultats publiés.",
        "Le sentier de randonnée serpentait à travers une forêt dense avant de s'ouvrir sur une vue spectaculaire de la vallée.",
        "La vie privée numérique reste un sujet controversé alors que les gouvernements tentent d'équilibrer sécurité et droits individuels.",
        "Les indicateurs économiques suggèrent que la reprise de la récession se poursuit plus rapidement que prévu par les analystes.",
        "Le chef a dressé le plat avec soin, ajoutant des micro-pousses et un filet d'huile de truffe pour la touche finale.",
        "Les réformateurs de l'éducation soutiennent que la pensée critique devrait être privilégiée par rapport à la mémorisation.",
    ],
    "german": [
        "Die Entwicklung erneuerbarer Energiequellen ist angesichts der wachsenden Bedenken zum Klimawandel zunehmend wichtig geworden.",
        "In einem kleinen Dorf zwischen sanften Hügeln versammelten sich die Bewohner zum jährlichen Erntedankfest.",
        "Die moderne Architektur legt oft Wert auf offene Grundrisse und nachhaltige Materialien, die die Umweltbelastung reduzieren.",
        "Das Orchester führte eine beeindruckende Aufführung der Neunten Symphonie von Beethoven auf, die das Publikum zu Begeisterung hinriss.",
        "Wissenschaftliche Forschung erfordert eine strenge Methodik und Peer-Review, um die Validität veröffentlichter Ergebnisse sicherzustellen.",
        "Der Wanderweg schlängelte sich durch dichten Wald, bevor er einen atemberaubenden Blick auf das Tal darunter freigab.",
        "Digitale Privatsphäre bleibt ein umstrittenes Thema, da Regierungen Sicherheitsbedenken mit individuellen Rechten abwägen.",
        "Wirtschaftsindikatoren deuten darauf hin, dass die Erholung von der Rezession schneller verläuft als ursprünglich prognostiziert.",
        "Der Küchenchef arrangierte das Gericht sorgfältig und fügte Mikrogreens und einen Schwall Trüffelöl als letzten Schliff hinzu.",
        "Bildungsreformer argumentieren, dass kritisches Denken gegenüber dem Auswendiglernen in Schulen Vorrang haben sollte.",
    ],
    "chinese": [
        "随着对气候变化担忧的加剧，可再生能源的开发变得越来越重要。",
        "在一个坐落在绵延山丘之间的小村庄里，居民们聚集在一起庆祝一年一度的丰收节。",
        "现代建筑通常强调开放式平面布局和可持续材料的使用，以减少对环境的影响。",
        "管弦乐团演奏了贝多芬第九交响曲的精彩版本，令观众起立鼓掌。",
        "科学研究需要严谨的方法论和同行评审，以确保已发表研究结果的有效性。",
        "徒步小径蜿蜒穿过茂密的森林，眼前豁然开朗，展现出山下山谷的壮丽景色。",
        "数字隐私仍然是一个有争议的问题，政府需要在安全关切与个人权利之间寻找平衡。",
        "经济指标表明，经济衰退的复苏速度比分析师最初预测的要快。",
        "厨师精心摆盘，添加了微型蔬菜和松露油的点睛之笔。",
        "教育改革者认为，学校应该优先培养批判性思维能力，而不是死记硬背。",
    ],
    "japanese": [
        "気候変動への懸念が高まる中、再生可能エネルギー源の開発がますます重要になっています。",
        "なだらかな丘に囲まれた小さな村で、住民たちが毎年の収穫祭に集まりました。",
        "現代の建築は、環境への影響を減らすオープンフロアプランや持続可能な素材を重視することが多いです。",
        "オーケストラはベートーヴェンの交響曲第9番の素晴らしい演奏を行い、観客をスタンディングオベーションさせました。",
        "科学研究は、発表された結果の妥当性を確保するために、厳格な方法論とピアレビューを必要とします。",
        "ハイキングコースは密集した森を通り抜け、眼下に広がる渓谷の息を呑むような景色が開けました。",
        "デジタルプライバシーは、政府が安全保障の懸念と個人の権利のバランスを取る中で、依然として議論の的となっています。",
        "経済指標は、景気後退からの回復がアナリストの当初の予測よりも速く進んでいることを示しています。",
        "シェフは丁寧に盛り付けし、仕上げにマイクログリーンとトリュフオイルを添えました。",
        "教育改革者は、暗記よりも批判的思考スキルを優先すべきだと主張しています。",
    ],
    "arabic": [
        "أصبح تطوير مصادر الطاقة المتجددة مهماً بشكل متزايد مع تزايد المخاوف بشأن تغير المناخ.",
        "في قرية صغيرة تقع بين تلال مموجة، اجتمع السكان للاحتفال بمهرجان الحصاد السنوي.",
        "غالباً ما تؤكد العمارة الحديثة على المخططات المفتوحة والمواد المستدامة التي تقلل التأثير البيئي.",
        "قدمت الأوركسترا أداءً مذهلاً للسمفونية التاسعة لبيتهوفن أثار الجمهور للوقوف تصفيقاً.",
        "يتطلب البحث العلمي منهجية صارمة ومراجعة الأقران لضمان صحة النتائج المنشورة.",
        "تعرج مسار المشي عبر غابة كثيفة قبل أن ينفتح على منظر خلاب لوادي الجبل.",
        "لا تزال الخصوصية الرقمية قضية مثيرة للجدل حيث توازن الحكومات بين المخاوف الأمنية والحقوق الفردية.",
        "تشير المؤشرات الاقتصادية إلى أن التعافي من الركود يسير بوتيرة أسرع مما توقعه المحللون.",
        "رتب الشيف الطبق بعناية، مضيفاً النباتات الدقيقة ورشة زيت الكمأة كلمسة نهائية.",
        "يجادل مصلحو التعليم بأن مهارات التفكير النقدي يجب أن تحظى بالأولوية على الحفظ في المدارس.",
    ],
    "hindi": [
        "जलवायु परिवर्तन की बढ़ती चिंताओं के बीच, नवीकरणीय ऊर्जा स्रोतों का विकास तेजी से महत्वपूर्ण होता जा रहा है।",
        "लहरदार पहाड़ियों के बीच बसे एक छोटे गांव में, निवासियों ने अपने वार्षिक फसल उत्सव के लिए इकट्ठा किया।",
        "आधुनिक वास्तुकला अक्सर खुले फ्लोर प्लान और पर्यावरण के अनुकूल सामग्री पर जोर देती है।",
        "ऑर्केस्ट्रा ने बीथोवन की नवीं सिम्फनी का एक शानदार प्रदर्शन किया जिससे दर्शक खड़े होकर ताली बजाने लगे।",
        "वैज्ञानिक अनुसंधान के लिए कड़ी कार्यप्रणाली और सहकर्मी समीक्षा आवश्यक है।",
        "ट्रैक घने जंगल से होकर गुजरता था और फिर नीचे घाटी के लुभावने दृश्य के सामने खुल जाता था।",
        "डिजिटल गोपनीयता एक विवादित मुद्दा बनी हुई है क्योंकि सरकारें सुरक्षा चिंताओं और व्यक्तिगत अधिकारों के बीच संतुलन खोजती हैं।",
        "आर्थिक संकेतक बताते हैं कि मंदी से उबार विश्लेषकों के प्रारंभिक अनुमान से तेजी से हो रहा है।",
        "शेफ ने व्यंजन को सावधानी से प्लेट किया और अंतिम स्पर्श के लिए माइक्रोग्रीन्स और ट्रफल ऑयल डाला।",
        "शिक्षा सुधारकों का तर्क है कि स्कूलों में रटने की तुलना में आलोचनात्मक सोच को प्राथमिकता दी जानी चाहिए।",
    ],
    "russian": [
        "Развитие возобновляемых источников энергии становится всё более важным на фоне растущих опасений по поводу изменения климата.",
        "В маленькой деревне, расположенной среди пологих холмов, жители собрались на ежегодный праздник урожая.",
        "Современная архитектура часто делает акцент на открытых планировках и экологически чистых материалах, снижающих воздействие на окружающую среду.",
        "Оркестр исполнил потрясающую версию Девятой симфонии Бетховена, заставив зрителей встать с мест.",
        "Научные исследования требуют строгой методологии и рецензирования для обеспечения достоверности публикуемых результатов.",
        "Тропа извивалась через густой лес, прежде чем открыться захватывающим видом на горную долину внизу.",
        "Цифровая конфиденциальность остаётся спорным вопросом, поскольку правительства пытаются сбалансировать проблемы безопасности и индивидуальные права.",
        "Экономические показатели показывают, что восстановление после рецессии идёт быстрее, чем первоначально прогнозировали аналитики.",
        "Шеф-повар аккуратно оформил блюдо, добавив микрозелень и каплю трюфельного масла в качестве финального штриха.",
        "Реформаторы образования утверждают, что навыки критического мышления должны иметь приоритет над зубрёжкой в школах.",
    ],
    "spanish": [
        "El desarrollo de fuentes de energía renovable se ha vuelto cada vez más importante ante la creciente preocupación por el cambio climático.",
        "En un pequeño pueblo entre colinas ondulantes, los residentes se reunieron para su festival anual de la cosecha.",
        "La arquitectura moderna a menudo enfatiza plantas abiertas y materiales sostenibles que reducen el impacto ambiental.",
        "La orquesta realizó una interpretación impresionante de la Novena Sinfonía de Beethoven que puso al público de pie.",
        "La investigación científica requiere una metodología rigurosa y revisión por pares para garantizar la validez de los resultados.",
        "El sendero serpenteaba a través de un denso bosque antes de abrirse a una vista impresionante del valle.",
        "La privacidad digital sigue siendo un tema controvertido mientras los gobiernos equilibran seguridad y derechos individuales.",
        "Los indicadores económicos sugieren que la recuperación de la recesión avanza más rápido de lo que los analistas predijeron.",
        "El chef emplató cuidadosamente el plato, añadiendo microbrotes y un chorrito de aceite de trufa como toque final.",
        "Los reformadores educativos argumentan que el pensamiento crítico debe priorizarse sobre la memorización en las escuelas.",
    ],
    "korean": [
        "기후 변화에 대한 우려가 커지면서 재생 에너지 원의 개발이 점점 더 중요해지고 있습니다.",
        "구릉진 언덕 사이에 자리 잡은 작은 마을에서 주민들이 연례 수확제를 위해 모였습니다.",
        "현대 건축은 환경 영향을 줄이는 개방형 평면 설계와 지속 가능한 자재를 강조하는 경우가 많습니다.",
        "오케스트라는 베토벤 교향곡 9번의 놀라운 연주를 선보여 관객들을 기립하게 만들었습니다.",
        "과학 연구는 출판된 연구 결과의 타당성을 보장하기 위해 엄격한 방법론과 동료 평가를 필요로 합니다.",
        "등산로는 울창한 숲을 지나며 계곡의 숨막히는 전망이 펼쳐졌습니다.",
        "디지털 프라이버시는 정부가 보안 우려와 개인 권리 사이의 균형을 맞추면서 여전히 논쟁거리입니다.",
        "경제 지표들은 침체에서의 회복이 분석가들이 처음 예상한 것보다 빠르게 진행되고 있음을 시사합니다.",
        "셰프가 조심스럽게 플레이팅하고 마지막 터치로 마이크로그린과 트러플 오일을 얹었습니다.",
        "교육 개혁가들은 학교에서 암기보다 비판적 사고 기술을 우선시해야 한다고 주장합니다.",
    ],
}


# Global accumulator for hook data
_hook_data = {}


def make_hook(layer_idx):
    def hook_fn(module, input, output):
        with torch.no_grad():
            logits = output[0] if isinstance(output, tuple) else output
            logits = logits.float()
            # top-1 expert for each token position
            top1 = torch.argmax(logits, dim=-1).cpu().numpy()
            # probability mass distribution
            probs = torch.softmax(logits, dim=-1)
            prob_sum = probs.sum(dim=0).cpu().numpy()  # sum across tokens
            _hook_data[layer_idx] = {
                "top1": top1,
                "prob_sum": prob_sum,
                "num_tokens": logits.shape[0] if logits.dim() == 2 else logits.shape[1],
            }
    return hook_fn


def main():
    parser = argparse.ArgumentParser(description="Test language-specific expert activation")
    parser.add_argument("--model-path", default="/home/kitt/gemma4-prune/bf16")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    from numpy.linalg import norm

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "22GiB", "cpu": "120GiB"},
        low_cpu_mem_usage=True,
    )

    if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
        text_model = model.model.language_model
        print("Found text model at model.model.language_model")
    elif hasattr(model, 'text_model'):
        text_model = model.text_model
    else:
        text_model = model

    config = model.config.text_config if hasattr(model.config, 'text_config') else model.config
    num_experts = getattr(config, 'num_experts', 128)
    num_layers = getattr(config, 'num_hidden_layers', 30)
    print(f"Model: {num_layers} layers, {num_experts} experts")

    try:
        processor = AutoProcessor.from_pretrained(args.model_path)
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Register hooks
    hooks = []
    for layer_idx, layer in enumerate(text_model.layers):
        if hasattr(layer, 'router'):
            hooks.append(layer.router.register_forward_hook(make_hook(layer_idx)))
    print(f"Registered {len(hooks)} hooks on routers")

    # Per-language accumulation
    lang_results = {}

    for lang_name, sentences in LANGUAGES.items():
        global _hook_data
        lang_top1 = np.zeros((num_layers, num_experts), dtype=np.int64)
        lang_prob = np.zeros((num_layers, num_experts), dtype=np.float64)
        lang_tokens = 0

        for text in sentences:
            _hook_data = {}  # reset for each sample
            messages = [{"role": "user", "content": text}]
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=128)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                _ = model(**inputs)

            # Accumulate hook data
            for layer_idx, data in _hook_data.items():
                for e in data["top1"].flatten():
                    lang_top1[layer_idx, e] += 1
                lang_prob[layer_idx] += data["prob_sum"]
                lang_tokens += data["num_tokens"]

        lang_results[lang_name] = {
            "top1": lang_top1,
            "prob": lang_prob,
            "tokens": lang_tokens,
        }
        print(f"  {lang_name:>10}: {lang_tokens:>6} tokens processed")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Clean up model
    del model, text_model
    torch.cuda.empty_cache()
    gc.collect()

    # ===== ANALYSIS =====
    print("\n" + "="*80)
    print("EXPERT ACTIVATION PATTERNS ACROSS LANGUAGES")
    print("="*80)

    all_langs = list(LANGUAGES.keys())

    # 1. Cosine similarity per layer (English vs others)
    print("\n--- Cosine similarity: English vs other languages (top-1 activation) ---")
    print(f"{'Layer':>6}", end="")
    for lang in all_langs[1:]:
        print(f"  {lang[:6]:>6}", end="")
    print()

    for layer_idx in range(num_layers):
        en_vec = lang_results["english"]["top1"][layer_idx].astype(np.float64)
        print(f"{layer_idx:>6}", end="")
        for lang in all_langs[1:]:
            lang_vec = lang_results[lang]["top1"][layer_idx].astype(np.float64)
            if norm(en_vec) > 0 and norm(lang_vec) > 0:
                sim = np.dot(en_vec, lang_vec) / (norm(en_vec) * norm(lang_vec))
            else:
                sim = 0.0
            print(f"  {sim:>6.3f}", end="")
        print()

    # 2. Average similarity per language
    print("\n--- Average similarity to English (across all layers) ---")
    avg_sims = {}
    for lang in all_langs[1:]:
        sims = []
        for layer_idx in range(num_layers):
            en_vec = lang_results["english"]["top1"][layer_idx].astype(np.float64)
            lang_vec = lang_results[lang]["top1"][layer_idx].astype(np.float64)
            if norm(en_vec) > 0 and norm(lang_vec) > 0:
                sim = np.dot(en_vec, lang_vec) / (norm(en_vec) * norm(lang_vec))
                sims.append(sim)
        avg_sims[lang] = np.mean(sims) if sims else 0.0
        print(f"  {lang:>10}: {avg_sims[lang]:.4f}")

    # 3. Language-specific experts
    print("\n--- Language-specific experts (non-English top-5, never in English top-5) ---")
    en_top5_all = set()
    non_en_top5_all = set()
    non_en_experts_by_lang = defaultdict(set)

    for layer_idx in range(num_layers):
        en_vec = lang_results["english"]["top1"][layer_idx]
        en_top5 = set(np.argsort(en_vec)[-5:].tolist())
        en_top5_all |= en_top5
        for lang in all_langs[1:]:
            lang_vec = lang_results[lang]["top1"][layer_idx]
            lang_top5 = set(np.argsort(lang_vec)[-5:].tolist())
            non_en_top5_all |= lang_top5
            for e in lang_top5:
                non_en_experts_by_lang[e].add(lang)

    language_specific = non_en_top5_all - en_top5_all
    shared = en_top5_all & non_en_top5_all
    english_only = en_top5_all - non_en_top5_all

    print(f"  Unique experts seen across all languages: {len(en_top5_all | non_en_top5_all)}")
    print(f"  English top-5 experts (any layer): {len(en_top5_all)}")
    print(f"  Non-English top-5 experts (any layer): {len(non_en_top5_all)}")
    print(f"  Shared (important for both EN and non-EN): {len(shared)}")
    print(f"  Language-specific (non-EN only): {len(language_specific)}")
    print(f"  English-only: {len(english_only)}")

    if language_specific:
        print(f"\n  Top language-specific experts:")
        sorted_specific = sorted(
            [(e, non_en_experts_by_lang[e]) for e in language_specific],
            key=lambda x: len(x[1]), reverse=True
        )[:15]
        for expert_id, langs in sorted_specific:
            print(f"    Expert {expert_id:>3}: {', '.join(sorted(langs))}")

    # 4. Conclusion
    overall_sim = np.mean(list(avg_sims.values())) if avg_sims else 0
    print(f"\n--- CONCLUSION ---")
    print(f"  Overall average cross-language similarity: {overall_sim:.4f}")
    if overall_sim > 0.95:
        print("  >> Expert routing is NEARLY IDENTICAL across languages.")
        print("  >> Language-based pruning won't differentiate experts effectively.")
        print("  >> Recommend: prune by overall activation frequency (calibration data).")
    elif overall_sim > 0.7:
        print("  >> Expert routing is MODERATELY SIMILAR across languages.")
        print("  >> Some language-specific experts exist. Pruning viable with care.")
    else:
        print("  >> Expert routing DIFFERS SIGNIFICANTLY across languages.")
        print("  >> Language-specific pruning is highly viable!")
        print("  >> Many experts are exclusively used by non-English languages.")

    # Save results
    output_path = Path("/home/kitt/gemma4-prune/lang_test_results.json")
    results_to_save = {}
    for lang, data in lang_results.items():
        results_to_save[lang] = {
            "top1": data["top1"].tolist(),
            "prob": data["prob"].tolist(),
            "tokens": int(data["tokens"]),
        }
    with open(output_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()