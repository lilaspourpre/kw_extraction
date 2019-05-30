from pymorphy2 import MorphAnalyzer

from tasks import *
from helpers import *

text = """Глава российского правительства Дмитрий Медведев подписал постановление, которое закрепляет отмену внутрисетевого роуминга на территории страны.
«Я подписал постановление, оно является развитием того, что было сделано до этого в законодательстве, а именно: закрепляет в правилах услуг телефонной связи отмену внутрисетевого роуминга», – сказал премьер, открывая традиционную встречу с вице-премьерами.
Медведев подчеркнул, что это окончательное решение вопроса на уровне подзаконного нормативного акта правительства, передает РИА «Новости».
В декабре президент России Владимир Путин подписал закон об отмене платы за национальный роуминг, обязав операторов мобильной связи устанавливать единые тарифы на все звонки, совершаемые по всей территории страны. Также сообщалось, что национальный роуминг будет отменен на территории Крыма.
При этом компании МТС, «Мегафон», «Вымпелком» и Tele2 – «большая четверка» мобильных операторов – были оштрафованы на 750 тыс. рублей каждая по делам о национальном (межсетевом) роуминге.
"""
morph = MorphAnalyzer()
normalized_text = normalize_text(text, morph)
joined_text = join_to_text([normalized_text])
tokens_with_ngrams = add_ngrams([normalized_text])


task1 = extract_kws_with_tfidf.delay(joined_text, 3)
task2 = extract_kws_with_scake.delay(tokens_with_ngrams, 10)
task3 = extract_kws_with_scake.delay(tokens_with_ngrams, 5)
task4 = extract_kws_with_tfidf.delay(joined_text, 10)

tasks = [task1, task2, task3, task4]

while tasks:
    for i in range(len(tasks)):
        if tasks[i].ready():
            print(tasks.pop(i).result)
            break

