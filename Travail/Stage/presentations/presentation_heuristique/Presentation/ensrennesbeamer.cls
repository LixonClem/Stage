\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ensrennesbeamer}[23/10/2017 Beamer style class for ENS Rennes]

\newif\if@info
\@infofalse
\DeclareOption{info}{\@infotrue}
\ProcessOptions\relax

\newif\if@dem
\@demfalse
\DeclareOption{dem}{\@demtrue}
\ProcessOptions\relax

\newif\if@mecatro
\@mecatrofalse
\DeclareOption{mecatro}{\@mecatrotrue}
\ProcessOptions\relax

\newif\if@sport
\@sportfalse
\DeclareOption{sport}{\@sporttrue}
\ProcessOptions\relax

\newif\if@math
\@mathfalse
\DeclareOption{math}{\@mathtrue}
\ProcessOptions\relax


\LoadClassWithOptions{beamer}

\RequirePackage[absolute,overlay]{textpos}
\RequirePackage{xcolor}

\if@info
  \definecolor{dptcolor}{HTML}{49bfe0}
\fi

\if@dem
  \definecolor{dptcolor}{HTML}{93117e}
\fi

\if@mecatro
  \definecolor{dptcolor}{HTML}{98bf0c}
\fi

\if@sport
  \definecolor{dptcolor}{HTML}{f29500}
\fi

\if@math
  \definecolor{dptcolor}{HTML}{40b4a6}
\fi

\RequirePackage{etoolbox}
\makeatletter
\patchcmd{\endbeamer@frameslide}{\ifx\beamer@frametitle\@empty}{\iffalse}{}{\errmessage{failed to patch}}
\makeatother

\useoutertheme{infolines}
\mode<presentation> {
  \usetheme{Warsaw}
  \definecolor{ensblue}{RGB}{50 76 152}
  \colorlet{mixcolor}{dptcolor!50!ensblue}
  \usefonttheme{default}
  \setbeamersize{mini frame size=0.8\paperwidth}
  \setbeamertemplate{navigation symbols}{}
  \setbeamertemplate{caption}[numbered]
  \setbeamercolor*{palette primary}{fg=white,bg=dptcolor}
  \setbeamercolor*{palette secondary}{fg=white,bg=mixcolor}
  \setbeamercolor*{section in head/foot}{fg=white, bg=ensblue}
  \setbeamercolor*{subsection in head/foot}{fg=white, bg=ensblue}
  \setbeamercolor*{author in head/foot}{fg=white, bg=ensblue}
  \setbeamercolor*{title}{fg=white, bg=ensblue}
  \setbeamercolor*{frametitle}{fg=white, bg=dptcolor}
  \setbeamercolor*{block title}{fg=white, bg=ensblue}
  \newcommand*\oldmacro{}%
  \let\oldmacro\insertshorttitle%
  \renewcommand*\insertshorttitle{%
    \oldmacro\hfill%
    \insertframenumber\,/\,\inserttotalframenumber}
  \setbeamertemplate{itemize items}[triangle]
  \addtobeamertemplate{frametitle}{}{
    \begin{textblock*}{100mm}(0.93\paperwidth ,0.01\textheight)
    \includegraphics[height=0.08\textheight]{logo_ens_blanc.png}
    \end{textblock*}
  }
  \addtobeamertemplate{frametitle}{}{
    \begin{textblock*}{100mm}(0cm,-0.1cm)
    \includegraphics[width=1.15\textwidth]{circles.png}
    \end{textblock*}
  }
}

\endinput