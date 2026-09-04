(function () {
  function initQuiz(quiz) {
    const answer = quiz.dataset.answer;
    const feedback = quiz.querySelector('.feedback');
    quiz.querySelectorAll('button[data-choice]').forEach((button) => {
      button.addEventListener('click', () => {
        const correct = button.dataset.choice === answer;
        feedback.textContent = correct
          ? (quiz.dataset.correct || '正确。先说出理由，再继续往下看。')
          : (quiz.dataset.wrong || '还不对。回到题干，找出它要求你比较的对象。');
        quiz.querySelectorAll('button[data-choice]').forEach((item) => {
          item.setAttribute('aria-pressed', item === button ? 'true' : 'false');
        });
      });
    });
  }

  function init() {
    document.querySelectorAll('.quiz').forEach(initQuiz);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
}());
