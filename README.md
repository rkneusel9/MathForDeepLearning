# MathForDeepLearning
Source code for the book "Math for Deep Learning"

Source code is organized by chapter.  If you have questions
or comments, please contact me:

rkneuselbooks@gmail.com

**Updates**
- p 300, the last sentence of the penultimate paragraph should read "Here, t, an integer starting at *one*, is the timestep."
- The file *boston.py* in Chapter 2 was sampling the same person repeatedly at times (thanks to ikimmit for the catch!)
- The file *tutorial.pdf* is a beginner's guide to NumPy, SciPy, Matplotlib, and Pillow.
- p 29, the upper limit on randint should be 365, not 364 (code updated).
- p 198, the derivative of a matrix function should be scalar $\partial x$, not $\partial\mathbf{x}$.
- p 257, the line above Equation 10.10 should be $\left[\frac{\partial E}{\partial y_0}\sigma'(x_0)\ \frac{\partial E}{\partial y_1}\sigma'(x_1)\ \ldots\ \right]^\top$.
- Tweaked the Ch 10 code in *build_dataset.py* to conform to newer Keras versions

