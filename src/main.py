from tasks import extract_kws_from_text

text = """Глава российского правительства Дмитрий Медведев подписал постановление, которое закрепляет отмену внутрисетевого роуминга на территории страны.
«Я подписал постановление, оно является развитием того, что было сделано до этого в законодательстве, а именно: закрепляет в правилах услуг телефонной связи отмену внутрисетевого роуминга», – сказал премьер, открывая традиционную встречу с вице-премьерами.
Медведев подчеркнул, что это окончательное решение вопроса на уровне подзаконного нормативного акта правительства, передает РИА «Новости».
В декабре президент России Владимир Путин подписал закон об отмене платы за национальный роуминг, обязав операторов мобильной связи устанавливать единые тарифы на все звонки, совершаемые по всей территории страны. Также сообщалось, что национальный роуминг будет отменен на территории Крыма.
При этом компании МТС, «Мегафон», «Вымпелком» и Tele2 – «большая четверка» мобильных операторов – были оштрафованы на 750 тыс. рублей каждая по делам о национальном (межсетевом) роуминге.
"""

task1 = extract_kws_from_text.delay(text, 3)
task2 = extract_kws_from_text.delay(text, 10)
task3 = extract_kws_from_text.delay(text, 5)

tasks = [task1, task2, task3]

while tasks:
    for i in range(len(tasks)):
        if tasks[i].ready():
            print(tasks.pop(i))
else:
    print(tasks)
    print(task1.result, task2.result, task3.result)


