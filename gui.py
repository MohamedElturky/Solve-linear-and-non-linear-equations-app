import math
import numpy as np
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from Bisection import Bisection
from FalsePosition import FalsePosition
from GaussElemination import GaussElemination
from Gaussjordan import GaussJordan
from LU import LU
from Jacobi import Jacobi
from GaussSeidel import GaussSeidel
from FixedPoint import FixedPoint
from Method import Equations 
from sympy import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from PyQt5.uic import loadUiType
from os import path 
import sys

from NewtonRaphson import NewtonRaphson
from Secant import Secant
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        
mainWindowFileName = "test.ui"                
FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), mainWindowFileName))
    

         

class Ui_MainWindow(QMainWindow,FORM_CLASS):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.system=Equations(0)
        self.scaling = False
        self.EnterNo.clicked.connect(self.setNoEqn)
        self.EnterEqn.clicked.connect(self.setEqn)
        self.FindSol.clicked.connect(self.findSol)
        self.Clear.clicked.connect(self.clear)
        self.EnterSigFigures.clicked.connect(self.setSigNo)
        self.method.currentTextChanged.connect(self.parameters)
        self.ScalingCheckBox.stateChanged.connect(self.toggleScaling)
        self.FindSolNonLin.clicked.connect(self.setNonLinEquation)
        self.drawGraph.clicked.connect(self.plot_expression)
        self.FindSolNonLin.clicked.connect(self.findSolNonLin)
        self.parameters()
        self.figure, self.ax = plt.subplots(figsize=(8, 6)) 

        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout(self.plotWidget)
        layout.addWidget(self.canvas)


        self.plotWidget.setLayout(layout)

       
        self.scrollAreaPlot.setWidgetResizable(True)

    def plot_expression(self):
        try:         
            expression = self.equation.toPlainText()
            if(self.is_valid_sympy_expression(expression)):

                x = symbols('x') 
                expr = sympify(expression)  

            
                x_vals = np.linspace(-100, 100, 5000)
                y_vals = []

                for val in x_vals:
                    try:
                    
                        y = float(expr.subs(x, val))
                        y_vals.append(y)
                    except (ZeroDivisionError, ValueError):
                        y_vals.append(np.nan)

                # Plot the graph
                self.ax.clear()
                self.ax.plot(x_vals, y_vals, label=str(expr))
                if(self.NonLinearMethod.currentText() == "Fixed Point"):
                    self.ax.plot(x_vals, x_vals, label="y = x", color="red")

                self.ax.axhline(0, color='black', linewidth=0.8) 
                self.ax.axvline(0, color='black', linewidth=0.8)  
                self.ax.grid(True)  
                self.ax.set_xlim(-10, 10)  
                self.ax.set_ylim(-10, 10)
                self.ax.set_title("Graph of the Expression")
                self.ax.set_xlabel("x")
                self.ax.set_ylabel("y")
                self.ax.legend()
                self.ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
                self.ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))
                self.ax.xaxis.set_minor_locator(AutoMinorLocator(4)) 
                self.ax.yaxis.set_minor_locator(AutoMinorLocator(4))

                self.ax.grid(which='both', linestyle='--', linewidth=0.5)  
                self.canvas.setMinimumSize(1000, 1000) 
            
                self.canvas.draw()

                # Add the toolbar
                if not hasattr(self, 'toolbar'):
                    self.toolbar = NavigationToolbar(self.canvas, self)
                    self.layout().addWidget(self.toolbar)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {e}")
    def toggleScaling(self, state):
        if state == QtCore.Qt.Checked:
            self.scaling = True

    def parameters (self):
        if(self.method.currentText()=="Gauss Elimination" or self.method.currentText()=="Gauss Jordan") :
            self.ParametersLabel.setEnabled(False)
            self.Parameters.setEnabled(False)
            self.InitialGuess.setEnabled(False)
            self.InitiaGuessLabel.setEnabled(False)
            self.IterationNumber.setEnabled(False)
            self.StoppingCondition.setEnabled(False)
            self.IterationLabel.setEnabled(False)
            self.StoppingConditionLabel.setEnabled(False)

        elif(self.method.currentText()=="LU decompostion") :
            self.ParametersLabel.setEnabled(True)
            self.Parameters.setEnabled(True)
            self.InitialGuess.setEnabled(False)
            self.InitiaGuessLabel.setEnabled(False)
            self.IterationNumber.setEnabled(False)
            self.StoppingCondition.setEnabled(False)
            self.IterationLabel.setEnabled(False)
            self.StoppingConditionLabel.setEnabled(False)

  
        elif(self.method.currentText()=="Jacobi" or self.method.currentText()=="Gauss sidel") :
            self.ParametersLabel.setEnabled(False)
            self.Parameters.setEnabled(False)
            self.InitialGuess.setEnabled(True)
            self.InitiaGuessLabel.setEnabled(True)
            self.IterationNumber.setEnabled(True)
            self.StoppingCondition.setEnabled(True)
            self.IterationLabel.setEnabled(True)
            self.StoppingConditionLabel.setEnabled(True)
    
    

    def is_number(self,s):
        try:
          float(s)
          return True
        except ValueError:
          return False
    def setNoEqn (self): 
      if(self.isWholeNumber(self.NoEqn.text())) :      
        eqnNo = int(self.NoEqn.text())
        self.No.setText("n= "+str(eqnNo)) 
        self.var.setText("a11 = ")
        self.system =Equations(eqnNo)
    def setSigNo (self): 
      if(self.isWholeNumber(self.SigFigures.text())) :      
        SigNo = int(self.SigFigures.text())
        self.system.sig = SigNo    
    
    def setEqn(self):
        if(self.Eqn.text()=="") :
            if(self.system.i<self.system.num):
                self.system.setCoff(0)
                self.Eqn.setText("")
                if(self.system.j < self.system.num and self.system.i < self.system.num):
                    self.var.setText(("a"+str(self.system.i+1)+str(self.system.j+1)+" ="))
                    
                elif(self.system.i < self.system.num) :               
                    self.var.setText("b"+str(self.system.i+1)+" =")
                    
                else:
                    self.var.setText("")

        elif(self.is_number(self.Eqn.text())) :
            if(self.system.i<self.system.num):
                self.system.setCoff(float(self.Eqn.text()))
                self.Eqn.setText("")
                if(self.system.j < self.system.num and self.system.i < self.system.num):
                    self.var.setText(("a"+str(self.system.i+1)+str(self.system.j+1)+" ="))
                    
                elif(self.system.i < self.system.num) :               
                    self.var.setText("b"+str(self.system.i+1)+" =")
                    
                else:
                    self.var.setText("")

    def findSol(self):
        if self.scaling:
            self.system.coff, self.system.sol = self.scale_matrix(self.system.coff,self.system.sol)

        print(self.system.coff)
        print(self.system.sol)
        if(self.method.currentText()=="Gauss Elimination"):
            A = np.array(self.system.coff)
            B = np.array(self.system.sol)
            gaussElem = GaussElemination(A,B,self.system.sig)
            startTime = time.time()
            self.result.setText("")
            self.view.setText(self.showEquations())
            try:                
                gaussElem.forwardElemination(0,0)
            except ValueError as e:
                self.result.setText(f"{e}")
            else:
                res = gaussElem.apply()
                EndTime = time.time()
                for i in range(len(res)):
                    self.result.setText(self.result.toPlainText()+f"X{i+1} = {res[i]}\n")
                self.time.setText(f"{EndTime - startTime}")
                

        
        elif(self.method.currentText()=="Gauss Jordan"):
            A = np.array(self.system.coff)
            B = np.array(self.system.sol)
            gaussJor = GaussJordan(A,B,self.system.sig)
            startTime = time.time()
            self.result.setText("")
            self.view.setText(self.showEquations())
            try:
                gaussJor.forwardElimination()
            except ValueError as e:
                self.result.setText(f"{e}")
            else:
                res = gaussJor.apply()
                EndTime = time.time()
                for i in range(len(res)):
                    self.result.setText(self.result.toPlainText()+f"X{i+1} = {res[i]}\n")
                self.time.setText(f"{EndTime - startTime}")

        elif(self.method.currentText()=="LU decompostion"):
            A = np.array(self.system.coff)
            B = np.array(self.system.sol)
            method = str(self.Parameters.currentText())
            lu = LU(A,B,method,self.system.sig)
            startTime = time.time()
            self.result.setText("")
            self.view.setText(self.showEquations())
            try:
                lu.forwardElemination(0,0)
                A = np.array(self.system.coff)
                B = np.array(self.system.sol)
                lu = LU(A,B,method,self.system.sig)
                res = lu.apply()
            except ValueError as e:
                self.result.setText(f"{e}")
            else:
                EndTime = time.time()
                for i in range(len(res)):
                    self.result.setText(self.result.toPlainText()+f"X{i+1} = {res[i]}\n")
                self.time.setText(f"{EndTime - startTime}")
        
        elif(self.method.currentText()=="Jacobi"):
            A = np.array(self.system.coff)
            B = np.array(self.system.sol)
            guess = self.InitialGuess.text()
            Guess = None if guess == "" else self.extractNumbers(guess) 
            iterations =self.IterationNumber.text()
            error = self.StoppingCondition.text()           
            if not(iterations == "" and error == ""):
                if(self.isWholeNumber(iterations) or iterations == "") and (self.is_number(error) or error == ""):
                    if(Guess is None or (all(isinstance(x, (int, float)) for x in Guess) and len(Guess) == len(self.system.sol))):
                        Iteration = None if iterations =="" else int(iterations)
                        Error = None if error =="" else float(error)
                        startTime = time.time()
                        self.result.setText("")
                        self.view.setText(self.showEquations())
                        try :
                            jacobi = Jacobi(A,B,Guess,Iteration,Error,self.system.sig)                           
                        except ValueError as e:
                            self.result.setText(f"{e}")
                        else :    
                            res,it = jacobi.apply()        
                            EndTime = time.time()
                            for i in range(len(res)):
                                self.result.setText(self.result.toPlainText()+f"X{i+1} = {res[i]}\n")
                            self.time.setText(f"{EndTime - startTime}")
                            self.Iterations.setText(f"{it}")     
        
        elif(self.method.currentText()=="Gauss sidel"):
            A = np.array(self.system.coff)
            B = np.array(self.system.sol)
            guess = self.InitialGuess.text()
            Guess = None if guess == "" else self.extractNumbers(guess) 
            iterations =self.IterationNumber.text()
            error = self.StoppingCondition.text()           
            if not(iterations == "" and error == ""):
                if(self.isWholeNumber(iterations) or iterations == "") and (self.is_number(error) or error == ""):
                    if(Guess is None or (all(isinstance(x, (int, float)) for x in Guess) and len(Guess) == len(self.system.sol))):
                        Iteration = None if iterations =="" else int(iterations)
                        Error = None if error =="" else float(error)
                        startTime = time.time()
                        self.result.setText("")
                        self.view.setText(self.showEquations())
                        try :
                            sd = GaussSeidel(A,B,Guess,Iteration,Error,self.system.sig)                           
                        except ValueError as e:
                            self.result.setText(f"{e}")
                        else :          
                            res,it = sd.apply()
                            EndTime = time.time()
                            for i in range(len(res)):
                                self.result.setText(self.result.toPlainText()+f"X{i+1} = {res[i]}\n")
                            self.time.setText(f"{EndTime - startTime}")
                            self.Iterations.setText(f"{it}")                               
    def findSolNonLin(self):
        if(self.NonLinearMethod.currentText()=="Fixed Point"):
            if(self.is_valid_sympy_expression(self.equation.toPlainText())):
                expression = self.equation.toPlainText()
                guess = self.NonLinearGuess.text()
                if(guess != "" and self.is_number(guess)):
                    x0 = float(guess)
                    tol = self.tolerance.text()
                    if((tol != "" and self.is_number(tol)) or tol == ""):
                        if(tol == ""):
                            tol = 1e-5
                        else:
                            tol = float(tol)
                        max_iter = self.iterations.text()
                        if((max_iter != "" and self.isWholeNumber(max_iter)) or max_iter == ""):
                            if(max_iter == ""):
                                max_iter = 50
                            else:
                                max_iter = int(max_iter)
                            sig = self.SigFigures.text()
                            if((sig != "" and self.isWholeNumber(sig)) or sig == ""):
                                if(sig == ""):
                                    sig = None
                                else:
                                    sig = int(sig)
                                step_by_step = self.stepsBox.isChecked()
                                fixed_point = FixedPoint(expression, x0, tol, max_iter, sig, step_by_step)
                                try:
                                    startTime = time.time()
                                    res, it, ea, step = fixed_point.apply()
                                    correctFig = floor(-int(math.log10(ea))) if ea != 0 else 0
                                except ValueError as e:
                                    QMessageBox.warning(self, "Error", f"An error occurred: {e}")
                                else:
                                    EndTime = time.time()
                                    self.result.setText(f"Root: {res}\nRelative Error: {ea*100}%\n")
                                    if correctFig != 0:
                                        self.result.setText(self.result.toPlainText() + f"Correct to {correctFig} significant figures")
                                    self.steps.setText(step)    
                                    self.Iterations.setText(f"{it}")
                                    self.time.setText(f"{EndTime - startTime}")
                            else:
                                QMessageBox.warning(self, "Validation Result", "Invalid number of significant figures!")
                        else:
                            QMessageBox.warning(self, "Validation Result", "Invalid maximum number of iterations!")
                    else:
                        QMessageBox.warning(self, "Validation Result", "Invalid tolerance!")
                else:
                    QMessageBox.warning(self, "Validation Result", "Invalid initial guess!")
            else:
                QMessageBox.warning(self, "Validation Result", "Invalid expression!")

        elif(self.NonLinearMethod.currentText()=="Secant"):
            if(self.is_valid_sympy_expression(self.equation.toPlainText())):
                expression = self.equation.toPlainText()
                guesses = self.NonLinearGuess.text().split(',')
                if len(guesses) == 2 and self.is_number(guesses[0]) and self.is_number(guesses[1]):
                    x0 = float(guesses[0])
                    x1 = float(guesses[1])
                    tol = self.tolerance.text()
                    if((tol != "" and self.is_number(tol)) or tol == ""):
                        if(tol == ""):
                            tol = 1e-5
                        else:
                            tol = float(tol)
                        max_iter = self.iterations.text()
                        if((max_iter != "" and self.isWholeNumber(max_iter)) or max_iter == ""):
                            if(max_iter == ""):
                                max_iter = 50
                            else:
                                max_iter = int(max_iter)
                            sig = self.SigFigures.text()
                            if((sig != "" and self.isWholeNumber(sig)) or sig == ""):
                                if(sig == ""):
                                    sig = None
                                else:
                                    sig = int(sig)
                                step_by_step = self.stepsBox.isChecked()
                                secant = Secant(expression, x0, x1, tol, max_iter, sig, step_by_step)
                                try:
                                    startTime = time.time()
                                    res, it, ea ,steps= secant.apply()
                                    correctFig = floor(-int(math.log10(ea))) if ea != 0 else 0
                                except ValueError as e:
                                    QMessageBox.warning(self, "Error", f"An error occurred: {e}")
                                else:
                                    EndTime = time.time()
                                    self.result.setText(f"Root: {res}\nRelative Error: {ea*100}%\n")
                                    if correctFig != 0:
                                        self.result.setText(self.result.toPlainText() + f"Correct to {correctFig} significant figures")
                                    self.steps.setText(steps)
                                    self.Iterations.setText(f"{it}")
                                    self.time.setText(f"{EndTime - startTime}")
                            else:
                                QMessageBox.warning(self, "Validation Result", "Invalid number of significant figures!")
                        else:
                            QMessageBox.warning(self, "Validation Result", "Invalid maximum number of iterations!")
                    else:
                        QMessageBox.warning(self, "Validation Result", "Invalid tolerance!")
                else:
                    QMessageBox.warning(self, "Validation Result", "Invalid initial guesses! Please enter x0 and x1 separated by a comma.")
            else:
                QMessageBox.warning(self, "Validation Result", "Invalid expression!")

        elif(self.NonLinearMethod.currentText()=="Bisection"):
            if(self.is_valid_sympy_expression(self.equation.toPlainText())):
                expression = self.equation.toPlainText()
                guesses = self.NonLinearGuess.text().split(',')
                if len(guesses) == 2 and self.is_number(guesses[0]) and self.is_number(guesses[1]):
                    x0 = float(guesses[0])
                    x1 = float(guesses[1])
                    tol = self.tolerance.text()
                    if((tol != "" and self.is_number(tol)) or tol == ""):
                        if(tol == ""):
                            tol = 1e-5
                        else:
                            tol = float(tol)
                        max_iter = self.iterations.text()
                        if((max_iter != "" and self.isWholeNumber(max_iter)) or max_iter == ""):
                            if(max_iter == ""):
                                max_iter = 50
                            else:
                                max_iter = int(max_iter)
                            sig = self.SigFigures.text()
                            if((sig != "" and self.isWholeNumber(sig)) or sig == ""):
                                if(sig == ""):
                                    sig = None
                                else:
                                    sig = int(sig)
                                step_by_step = self.stepsBox.isChecked()
                                bisection = Bisection(expression, x0, x1, tol, max_iter, sig, step_by_step)
                                try:
                                    startTime = time.time()
                                    res, it, ea ,steps= bisection.solve()
                                    correctFig = floor(-int(math.log10(ea))) if ea != 0 else 0
                                except ValueError as e:
                                    QMessageBox.warning(self, "Error", f"An error occurred: {e}")
                                else:
                                    EndTime = time.time()
                                    self.result.setText(f"Root: {res}\nRelative Error:  {ea*100}%\n")
                                    if correctFig != 0:
                                        self.result.setText(self.result.toPlainText() + f"Correct to {correctFig} significant figures")
                                    self.Iterations.setText(f"{it}")
                                    self.steps.setText(steps)
                                    self.time.setText(f"{EndTime - startTime}")
                            else:
                                QMessageBox.warning(self, "Validation Result", "Invalid number of significant figures!")
                        else:
                            QMessageBox.warning(self, "Validation Result", "Invalid maximum number of iterations!")
                    else:
                        QMessageBox.warning(self, "Validation Result", "Invalid tolerance!")
                else:
                    QMessageBox.warning(self, "Validation Result", "Invalid initial guesses! Please enter x0 and x1 separated by a comma.")
            else:
                QMessageBox.warning(self, "Validation Result", "Invalid expression!")

        elif(self.NonLinearMethod.currentText()=="False-Position"):
            if(self.is_valid_sympy_expression(self.equation.toPlainText())):
                expression = self.equation.toPlainText()
                guesses = self.NonLinearGuess.text().split(',')
                if len(guesses) == 2 and self.is_number(guesses[0]) and self.is_number(guesses[1]):
                    x0 = float(guesses[0])
                    x1 = float(guesses[1])
                    tol = self.tolerance.text()
                    if((tol != "" and self.is_number(tol)) or tol == ""):
                        if(tol == ""):
                            tol = 1e-5
                        else:
                            tol = float(tol)
                        max_iter = self.iterations.text()
                        if((max_iter != "" and self.isWholeNumber(max_iter)) or max_iter == ""):
                            if(max_iter == ""):
                                max_iter = 50
                            else:
                                max_iter = int(max_iter)
                            sig = self.SigFigures.text()
                            if((sig != "" and self.isWholeNumber(sig)) or sig == ""):
                                if(sig == ""):
                                    sig = None
                                else:
                                    sig = int(sig)
                                step_by_step = self.stepsBox.isChecked()
                                false_position = FalsePosition(expression, x0, x1, tol, max_iter, sig, step_by_step)
                                try:
                                    startTime = time.time()
                                    res, it, ea ,steps= false_position.solve()
                                    correctFig = floor(-int(math.log10(ea))) if ea != 0 else 0
                                except ValueError as e:
                                    QMessageBox.warning(self, "Error", f"An error occurred: {e}")
                                else:
                                    EndTime = time.time()
                                    self.result.setText(f"Root: {res}\nRelative Error:  {ea*100}%\n")
                                    if correctFig != 0:
                                        self.result.setText(self.result.toPlainText() + f"Correct to {correctFig} significant figures")
                                    self.Iterations.setText(f"{it}")
                                    self.steps.setText(steps)
                                    self.time.setText(f"{EndTime - startTime}")
                            else:
                                QMessageBox.warning(self, "Validation Result", "Invalid number of significant figures!")
                        else:
                            QMessageBox.warning(self, "Validation Result", "Invalid maximum number of iterations!")
                    else:
                        QMessageBox.warning(self, "Validation Result", "Invalid tolerance!")
                else:
                    QMessageBox.warning(self, "Validation Result", "Invalid initial guesses! Please enter x0 and x1 separated by a comma.")
            else:
                QMessageBox.warning(self, "Validation Result", "Invalid expression!")

        elif(self.NonLinearMethod.currentText()=="Original Netwon Raphson" or self.NonLinearMethod.currentText()=="Modified Newton Raphson"):
            if(self.NonLinearMethod.currentText()=="Original Netwon Raphson"):
                m = 1
                modified = False
            else:
                modified = True 
                m = self.mult.text()
            if(m == "" or(m!="" and self.isWholeNumber(m))):
                m = None if m == "" else int(m)    
                if(self.is_valid_sympy_expression(self.equation.toPlainText())):
                    expression = self.equation.toPlainText()
                    guess = self.NonLinearGuess.text()
                    if((self.SigFigures.text() != "" and self.isWholeNumber(self.SigFigures.text()) ) or self.SigFigures.text() == ""): 
                        if(self.SigFigures.text() == ""):
                            sig = None
                        else:
                            sig = self.SigFigures.text()
                            sig = int(sig)
                        if(guess != "" and self.is_number(guess)):
                            x0 = float(guess)
                            tol = self.tolerance.text()
                            if((tol != "" and self.is_number(tol)) or tol == ""):
                                if(tol == ""):
                                    tol = 1e-5
                                else:   
                                    tol = float(tol)
                                max_iter = self.iterations.text()
                                if((max_iter != "" and self.isWholeNumber(max_iter)) or max_iter == ""):
                                    if(max_iter == ""):
                                        max_iter = 50
                                    else:
                                        max_iter = int(max_iter)
                                    step_by_step = self.stepsBox.isChecked()
                                    newton_raphson = NewtonRaphson(expression, x0,modified, m, tol, sig, max_iter, step_by_step)
                                    try:
                                        startTime = time.time()
                                        res, it, ea,steps = newton_raphson.solve()
                                        correctFig = floor(-int(math.log10(ea))) if ea != 0 else 0
                                    except ValueError as e:
                                        QMessageBox.warning(self, "Error", f"An error occurred: {e}")
                                    else:
                                        EndTime = time.time()
                                        self.result.setText(f"Root: {res}\nRelative Error:  {ea*100}%\n")
                                        if correctFig !=0:
                                            self.result.setText(self.result.toPlainText()+f"Correct to {correctFig} significant figures") 
                                        self.steps.setText(steps)
                                        self.Iterations.setText(f"{it}")
                                        self.time.setText(f"{EndTime - startTime}")
                                else:
                                    QMessageBox.warning(self, "Validation Result", "Invalid maximum number of iterations!")
                            else:
                                QMessageBox.warning(self, "Validation Result", "Invalid tolerance!")
                        else:
                            QMessageBox.warning(self, "Validation Result", "Invalid initial guess!")
                    else:
                        QMessageBox.warning(self, "Validation Result", "Invalid number of significant figures!")    
                else:
                    QMessageBox.warning(self, "Validation Result", "Invalid expression!")
            else:
                QMessageBox.warning(self, "Validation Result", "Invalid value for m!") 
    def scale_matrix(self, A, B):
        n = A.shape[0]
        for i in range(n):
            row_max = max(abs(A[i, j]) for j in range(n))
            if row_max != 0:
                A[i, :] = A[i, :] / row_max
                B[i] = B[i] / row_max
        return A, B



    def clear(self):
        self.system=Equations(0)
        self.No.setText("n=")
        self.NoEqn.setText("")
        self.var.setText("")
        self.result.setText("")
        self.Eqn.setText("")
        self.InitialGuess.setText("")
        self.IterationNumber.setText("")
        self.StoppingCondition.setText("")
        self.time.setText("")
        self.Iterations.setText("")
        self.SigFigures.setText("")
        self.result.setText("")
        self.view.setText("")
    
    def isWholeNumber(self, s):
        try:
            number = float(s)  
            if number % 1 == 0: 
                return True
            else:
                return False
        except ValueError:
            return False
    def extractNumbers(self, s):
        try:
            numbers = s.split(',')
            numbers = [float(num) for num in numbers]           
            return numbers
        except ValueError:
            return []
    def showEquations(self):
        eqn =  ""
        for i in range(len(self.system.sol)):
            for j in range(len(self.system.sol)):
                if j == len(self.system.sol)-1:
                    eqn += f"({self.system.coff[i,j]}) X{j+1} ="
                else:
                    eqn += f"({self.system.coff[i,j]}) X{j+1} +"
            eqn +=  f"{self.system.sol[i]}\n"
        return eqn      
    def setNonLinEquation(self):
        if(self.equation.toPlainText() != ""):
            if(self.is_valid_sympy_expression(self.equation.toPlainText())):
                
                expr = self.equation.toPlainText()
                init_printing()
                print(expr)
            else:
                QMessageBox.warning(self, "Validation Result", "Invalid variable!")     

    def is_valid_sympy_expression(self, expression):
        try:
            x = symbols('x')
            expr = sympify(expression)
            return expr.free_symbols.issubset({x})
            
        except SympifyError:
            QMessageBox.warning(self, "Validation Result", "Invalid expression!")
            return False        
    
def loadStylesheet():
    try:
        with open("StyleSheet.qss", "r") as file:
            stylesheet = file.read()
            QApplication.instance().setStyleSheet(stylesheet)
    except FileNotFoundError:
        print("Stylesheet file not found.")


def main():
    app = QApplication(sys.argv)
    loadStylesheet()
    window = Ui_MainWindow()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()
